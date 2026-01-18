from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
class VmwareContentDeployOvfTemplate(VmwareRestClient):

    def __init__(self, module):
        """Constructor."""
        super(VmwareContentDeployOvfTemplate, self).__init__(module)
        self.module = module
        self._pyv = PyVmomi(module=module)
        self._template_service = self.api_client.vcenter.vm_template.LibraryItems
        self._datacenter_id = None
        self._datastore_id = None
        self._library_item_id = None
        self._folder_id = None
        self._host_id = None
        self._cluster_id = None
        self._resourcepool_id = None
        self.result = {}
        if self.module._debug:
            self.warn('Enable debug output because ANSIBLE_DEBUG was set.')
            self.params['log_level'] = 'debug'
        self.log_level = self.params['log_level']
        if self.log_level == 'debug':
            self.result['debug'] = {}
        self.template = self.params.get('template')
        self.library = self.params.get('library')
        self.vm_name = self.params.get('name')
        self.datacenter = self.params.get('datacenter')
        self.datastore = self.params.get('datastore')
        self.datastore_cluster = self.params.get('datastore_cluster')
        self.folder = self.params.get('folder')
        self.resourcepool = self.params.get('resource_pool')
        self.cluster = self.params.get('cluster')
        self.host = self.params.get('host')
        self.storage_provisioning = self.params['storage_provisioning']
        if self.storage_provisioning == 'eagerzeroedthick':
            self.storage_provisioning = 'eagerZeroedThick'
        vm = self._pyv.get_vm()
        if vm:
            self.result['vm_deploy_info'] = dict(msg="Virtual Machine '%s' already Exists." % self.vm_name, vm_id=vm._moId)
            self._fail(msg='Virtual Machine deployment failed')

    def deploy_vm_from_ovf_template(self):
        self._datacenter_id = self.get_datacenter_by_name(self.datacenter)
        if not self._datacenter_id:
            self._fail(msg='Failed to find the datacenter %s' % self.datacenter)
        if self.datastore:
            self._datastore_id = self.get_datastore_by_name(self.datacenter, self.datastore)
            if not self._datastore_id:
                self._fail(msg='Failed to find the datastore %s' % self.datastore)
        if self.datastore_cluster and (not self._datastore_id):
            dsc = self._pyv.find_datastore_cluster_by_name(self.datastore_cluster)
            if dsc:
                self.datastore = self._pyv.get_recommended_datastore(dsc)
                self._datastore_id = self.get_datastore_by_name(self.datacenter, self.datastore)
            else:
                self._fail(msg='Failed to find the datastore cluster %s' % self.datastore_cluster)
        if not self._datastore_id:
            self._fail(msg='Failed to find the datastore using either datastore or datastore cluster')
        if self.library:
            self._library_item_id = self.get_library_item_from_content_library_name(self.template, self.library)
            if not self._library_item_id:
                self._fail(msg='Failed to find the library Item %s in content library %s' % (self.template, self.library))
        else:
            self._library_item_id = self.get_library_item_by_name(self.template)
            if not self._library_item_id:
                self._fail(msg='Failed to find the library Item %s' % self.template)
        folder_obj = self._pyv.find_folder_by_fqpn(self.folder, self.datacenter, folder_type='vm')
        if folder_obj:
            self._folder_id = folder_obj._moId
        if not self._folder_id:
            self._fail(msg='Failed to find the folder %s' % self.folder)
        if self.host:
            self._host_id = self.get_host_by_name(self.datacenter, self.host)
            if not self._host_id:
                self._fail(msg='Failed to find the Host %s' % self.host)
        if self.cluster:
            self._cluster_id = self.get_cluster_by_name(self.datacenter, self.cluster)
            if not self._cluster_id:
                self._fail(msg='Failed to find the Cluster %s' % self.cluster)
            cluster_obj = self.api_client.vcenter.Cluster.get(self._cluster_id)
            self._resourcepool_id = cluster_obj.resource_pool
        if self.resourcepool:
            self._resourcepool_id = self.get_resource_pool_by_name(self.datacenter, self.resourcepool, self.cluster, self.host)
            if not self._resourcepool_id:
                self._fail(msg='Failed to find the resource_pool %s' % self.resourcepool)
        if not self._resourcepool_id:
            self._fail(msg='Failed to find a resource pool either by name or cluster')
        deployment_target = LibraryItem.DeploymentTarget(resource_pool_id=self._resourcepool_id, folder_id=self._folder_id)
        self.ovf_summary = self.api_client.vcenter.ovf.LibraryItem.filter(ovf_library_item_id=self._library_item_id, target=deployment_target)
        self.deploy_spec = LibraryItem.ResourcePoolDeploymentSpec(name=self.vm_name, annotation=self.ovf_summary.annotation, accept_all_eula=True, network_mappings=None, storage_mappings=None, storage_provisioning=self.storage_provisioning, storage_profile_id=None, locale=None, flags=None, additional_parameters=None, default_datastore_id=self._datastore_id)
        response = {'succeeded': False}
        try:
            response = self.api_client.vcenter.ovf.LibraryItem.deploy(self._library_item_id, deployment_target, self.deploy_spec)
        except Error as error:
            self._fail(msg='%s' % self.get_error_message(error))
        except Exception as err:
            self._fail(msg='%s' % to_native(err))
        if not response.succeeded:
            self.result['vm_deploy_info'] = dict(msg='Virtual Machine deployment failed', vm_id='')
            self._fail(msg='Virtual Machine deployment failed')
        self.result['changed'] = True
        self.result['vm_deploy_info'] = dict(msg="Deployed Virtual Machine '%s'." % self.vm_name, vm_id=response.resource_id.id)
        self._exit()

    def _mod_debug(self):
        if self.log_level == 'debug':
            self.result['debug'].update(dict(datacenter_id=self._datacenter_id, datastore_id=self._datastore_id, library_item_id=self._library_item_id, folder_id=self._folder_id, host_id=self._host_id, cluster_id=self._cluster_id, resourcepool_id=self._resourcepool_id))

    def _fail(self, msg):
        self._mod_debug()
        self.module.fail_json(msg=msg, **self.result)

    def _exit(self):
        self._mod_debug()
        self.module.exit_json(**self.result)