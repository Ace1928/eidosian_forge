from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVirtualMachineScaleSetVMInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), vmss_name=dict(type='str', required=True), instance_id=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.vmss_name = None
        self.instance_id = None
        self.tags = None
        super(AzureRMVirtualMachineScaleSetVMInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_virtualmachinescalesetinstance_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_virtualmachinescalesetinstance_facts' module has been renamed to" + " 'azure_rm_virtualmachinescalesetinstance_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(ComputeManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-04-01')
        if self.instance_id is None:
            self.results['instances'] = self.list()
        else:
            self.results['instances'] = self.get()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.virtual_machine_scale_set_vms.get(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, instance_id=self.instance_id)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Virtual Machine Scale Set VM.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list(self):
        items = None
        try:
            items = self.mgmt_client.virtual_machine_scale_set_vms.list(resource_group_name=self.resource_group, virtual_machine_scale_set_name=self.vmss_name)
            self.log('Response : {0}'.format(items))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Virtual Machine ScaleSet VM.')
        results = []
        for item in items:
            if self.has_tags(item.tags, self.tags):
                results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        iv = self.mgmt_client.virtual_machine_scale_set_vms.get_instance_view(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, instance_id=d.get('instance_id', None)).as_dict()
        power_state = ''
        for index in range(len(iv['statuses'])):
            code = iv['statuses'][index]['code'].split('/')
            if code[0] == 'PowerState':
                power_state = code[1]
                break
        d = {'resource_group': self.resource_group, 'id': d.get('id', None), 'tags': d.get('tags', None), 'instance_id': d.get('instance_id', None), 'latest_model': d.get('latest_model_applied', None), 'name': d.get('name', None), 'provisioning_state': d.get('provisioning_state', None), 'power_state': power_state, 'vm_id': d.get('vm_id', None), 'image_reference': d.get('storage_profile').get('image_reference', None), 'computer_name': d.get('os_profile').get('computer_name', None)}
        return d