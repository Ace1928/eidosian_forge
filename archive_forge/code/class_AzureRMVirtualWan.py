from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMVirtualWan(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(location=dict(type='str'), resource_group=dict(type='str', required=True), office365_local_breakout_category=dict(type='str', choices=['Optimize', 'OptimizeAndAllow', 'All', 'None']), name=dict(type='str', required=True), disable_vpn_encryption=dict(type='bool', disposition='/disable_vpn_encryption'), virtual_hubs=dict(type='list', elements='dict', updatable=False, disposition='/virtual_hubs', options=dict(id=dict(type='str', disposition='id'))), vpn_sites=dict(type='list', elements='dict', updatable=False, disposition='/vpn_sites', options=dict(id=dict(type='str', disposition='id'))), allow_branch_to_branch_traffic=dict(type='bool', disposition='/allow_branch_to_branch_traffic'), allow_vnet_to_vnet_traffic=dict(type='bool', updatable=False, disposition='/allow_vnet_to_vnet_traffic'), virtual_wan_type=dict(type='str', disposition='/virtual_wan_type', choices=['Basic', 'Standard']), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.location = None
        self.body = {}
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMVirtualWan, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        resource_group = self.get_resource_group(self.resource_group)
        if self.location is None:
            self.location = resource_group.location
        self.body['location'] = self.location
        old_response = None
        response = None
        old_response = self.get_resource()
        if not old_response:
            if self.state == 'present':
                self.to_do = Actions.Create
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            modifiers = {}
            self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
            self.results['modifiers'] = modifiers
            self.results['compare'] = []
            if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_resource()
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
        else:
            self.results['changed'] = False
            response = old_response
        if response is not None:
            self.results['state'] = response
        return self.results

    def create_update_resource(self):
        try:
            response = self.network_client.virtual_wans.begin_create_or_update(resource_group_name=self.resource_group, virtual_wan_name=self.name, wan_parameters=self.body)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the VirtualWan instance.')
            self.fail('Error creating the VirtualWan instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_resource(self):
        try:
            response = self.network_client.virtual_wans.begin_delete(resource_group_name=self.resource_group, virtual_wan_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the VirtualWan instance.')
            self.fail('Error deleting the VirtualWan instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        try:
            response = self.network_client.virtual_wans.get(resource_group_name=self.resource_group, virtual_wan_name=self.name)
        except ResourceNotFoundError as e:
            return False
        return response.as_dict()