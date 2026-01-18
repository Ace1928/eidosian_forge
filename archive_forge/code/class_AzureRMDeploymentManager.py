from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDeploymentManager(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True, aliases=['resource_group_name']), name=dict(type='str', default='ansible-arm', aliases=['deployment_name']), state=dict(type='str', default='present', choices=['present', 'absent']), template=dict(type='dict', default=None), parameters=dict(type='dict', default=None), template_link=dict(type='str', default=None), parameters_link=dict(type='str', default=None), location=dict(type='str', default='westus'), deployment_mode=dict(type='str', default='incremental', choices=['complete', 'incremental']), wait_for_deployment_completion=dict(type='bool', default=True), wait_for_deployment_polling_period=dict(type='int', default=10))
        mutually_exclusive = [('template', 'template_link'), ('parameters', 'parameters_link')]
        self.resource_group = None
        self.state = None
        self.template = None
        self.parameters = None
        self.template_link = None
        self.parameters_link = None
        self.location = None
        self.deployment_mode = None
        self.name = None
        self.wait_for_deployment_completion = None
        self.wait_for_deployment_polling_period = None
        self.tags = None
        self.append_tags = None
        self.results = dict(deployment=dict(), changed=False, msg='')
        super(AzureRMDeploymentManager, self).__init__(derived_arg_spec=self.module_arg_spec, mutually_exclusive=mutually_exclusive, supports_check_mode=False)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['append_tags', 'tags']:
            setattr(self, key, kwargs[key])
        if self.state == 'present':
            deployment = self.deploy_template()
            if deployment is None:
                self.results['deployment'] = dict(name=self.name, group_name=self.resource_group, id=None, outputs=None, instances=None)
            else:
                self.results['deployment'] = dict(name=deployment.name, group_name=self.resource_group, id=deployment.id, outputs=deployment.properties.outputs, instances=self._get_instances(deployment))
            self.results['changed'] = True
            self.results['msg'] = 'deployment succeeded'
        else:
            try:
                if self.get_resource_group(self.resource_group):
                    self.destroy_resource_group()
                    self.results['changed'] = True
                    self.results['msg'] = 'deployment deleted'
            except Exception:
                pass
        return self.results

    def deploy_template(self):
        """
        Deploy the targeted template and parameters
        :param module: Ansible module containing the validated configuration for the deployment template
        :param client: resource management client for azure
        :param conn_info: connection info needed
        :return:
        """
        deploy_parameter = self.rm_models.DeploymentProperties(mode=self.deployment_mode)
        if not self.parameters_link:
            deploy_parameter.parameters = self.parameters
        else:
            deploy_parameter.parameters_link = self.rm_models.ParametersLink(uri=self.parameters_link)
        if not self.template_link:
            deploy_parameter.template = self.template
        else:
            deploy_parameter.template_link = self.rm_models.TemplateLink(uri=self.template_link)
        try:
            rg = self.rm_client.resource_groups.get(self.resource_group)
            if rg.tags:
                update_tags, self.tags = self.update_tags(rg.tags)
        except ResourceNotFoundError:
            pass
        params = self.rm_models.ResourceGroup(location=self.location, tags=self.tags)
        try:
            self.rm_client.resource_groups.create_or_update(self.resource_group, params)
        except Exception as exc:
            self.fail('Resource group create_or_update failed with status code: %s and message: %s' % (exc.status_code, exc.message))
        try:
            result = self.rm_client.deployments.begin_create_or_update(self.resource_group, self.name, {'properties': deploy_parameter})
            deployment_result = None
            if self.wait_for_deployment_completion:
                deployment_result = self.get_poller_result(result)
                while deployment_result.properties is None or deployment_result.properties.provisioning_state not in ['Canceled', 'Failed', 'Deleted', 'Succeeded']:
                    time.sleep(self.wait_for_deployment_polling_period)
                    deployment_result = self.rm_client.deployments.get(self.resource_group, self.name)
        except Exception as exc:
            failed_deployment_operations = self._get_failed_deployment_operations(self.name)
            self.log('Deployment failed %s: %s' % (exc.status_code, exc.message))
            error_msg = self._error_msg_from_cloud_error(exc)
            self.fail(error_msg, failed_deployment_operations=failed_deployment_operations)
        if self.wait_for_deployment_completion and deployment_result.properties.provisioning_state != 'Succeeded':
            self.log('provisioning state: %s' % deployment_result.properties.provisioning_state)
            failed_deployment_operations = self._get_failed_deployment_operations(self.name)
            self.fail('Deployment failed. Deployment id: %s' % deployment_result.id, failed_deployment_operations=failed_deployment_operations)
        return deployment_result

    def destroy_resource_group(self):
        """
        Destroy the targeted resource group
        """
        try:
            result = self.rm_client.resource_groups.begin_delete(self.resource_group)
            result.wait()
        except Exception as e:
            if e.status_code == 404 or e.status_code == 204:
                return
            else:
                self.fail('Delete resource group and deploy failed with status code: %s and message: %s' % (e.status_code, e.message))

    def _get_failed_nested_operations(self, current_operations):
        new_operations = []
        for operation in current_operations:
            if operation.properties.provisioning_state == 'Failed':
                new_operations.append(operation)
                if operation.properties.target_resource and 'Microsoft.Resources/deployments' in operation.properties.target_resource.id:
                    nested_deployment = operation.properties.target_resource.resource_name
                    try:
                        nested_operations = self.rm_client.deployment_operations.list(self.resource_group, nested_deployment)
                    except Exception as exc:
                        self.fail('List nested deployment operations failed with status code: %s and message: %s' % (exc.status_code, exc.message))
                    new_nested_operations = self._get_failed_nested_operations(nested_operations)
                    new_operations += new_nested_operations
        return new_operations

    def _get_failed_deployment_operations(self, name):
        results = []
        try:
            operations = self.rm_client.deployment_operations.list(self.resource_group, name)
        except Exception as exc:
            self.fail('Get deployment failed with status code: %s and message: %s' % (exc.status_code, exc.message))
        try:
            results = [dict(id=op.id, operation_id=op.operation_id, status_code=op.properties.status_code, status_message=op.properties.status_message, target_resource=dict(id=op.properties.target_resource.id, resource_name=op.properties.target_resource.resource_name, resource_type=op.properties.target_resource.resource_type) if op.properties.target_resource else None, provisioning_state=op.properties.provisioning_state) for op in self._get_failed_nested_operations(operations)]
        except Exception:
            pass
        self.log(dict(failed_deployment_operations=results), pretty_print=True)
        return results

    def _get_instances(self, deployment):
        dep_tree = self._build_hierarchy(deployment.properties.dependencies)
        vms = self._get_dependencies(dep_tree, resource_type='Microsoft.Compute/virtualMachines')
        vms_and_nics = [(vm, self._get_dependencies(vm['children'], 'Microsoft.Network/networkInterfaces')) for vm in vms]
        vms_and_ips = [(vm['dep'], self._nic_to_public_ips_instance(nics)) for vm, nics in vms_and_nics]
        return [dict(vm_name=vm.resource_name, ips=[self._get_ip_dict(ip) for ip in ips]) for vm, ips in vms_and_ips if len(ips) > 0]

    def _get_dependencies(self, dep_tree, resource_type):
        matches = [value for value in dep_tree.values() if value['dep'].resource_type == resource_type]
        for child_tree in [value['children'] for value in dep_tree.values()]:
            matches += self._get_dependencies(child_tree, resource_type)
        return matches

    def _build_hierarchy(self, dependencies, tree=None):
        tree = dict(top=True) if tree is None else tree
        for dep in dependencies:
            if dep.resource_name not in tree:
                tree[dep.resource_name] = dict(dep=dep, children=dict())
            if isinstance(dep, self.rm_models.Dependency) and dep.depends_on is not None and (len(dep.depends_on) > 0):
                self._build_hierarchy(dep.depends_on, tree[dep.resource_name]['children'])
        if 'top' in tree:
            tree.pop('top', None)
            keys = list(tree.keys())
            for key1 in keys:
                for key2 in keys:
                    if key2 in tree and key1 in tree[key2]['children'] and (key1 in tree):
                        tree[key2]['children'][key1] = tree[key1]
                        tree.pop(key1)
        return tree

    def _get_ip_dict(self, ip):
        ip_dict = dict(name=ip.name, id=ip.id, public_ip=ip.ip_address, public_ip_allocation_method=str(ip.public_ip_allocation_method))
        if ip.dns_settings:
            ip_dict['dns_settings'] = {'domain_name_label': ip.dns_settings.domain_name_label, 'fqdn': ip.dns_settings.fqdn}
        return ip_dict

    def _nic_to_public_ips_instance(self, nics):
        nic_list = []
        for nic in nics:
            resp = None
            try:
                resp = self.network_client.network_interfaces.get(self.resource_group, nic['dep'].resource_name)
            except ResourceNotFoundError:
                pass
            if resp is not None:
                nic_list.append(resp)
        return [self.network_client.public_ip_addresses.get(public_ip_id.split('/')[4], public_ip_id.split('/')[-1]) for nic_obj in nic_list for public_ip_id in [ip_conf_instance.public_ip_address.id for ip_conf_instance in nic_obj.ip_configurations if ip_conf_instance.public_ip_address]]

    def _error_msg_from_cloud_error(self, exc):
        msg = ''
        status_code = str(exc.status_code)
        if status_code.startswith('2'):
            msg = 'Deployment failed: {0}'.format(exc.message)
        else:
            msg = 'Deployment failed with status code: {0} and message: {1}'.format(status_code, exc.message)
        return msg