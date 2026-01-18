from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMVpnSite(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), virtual_wan=dict(type='dict', disposition='/virtual_wan', options=dict(id=dict(type='str', disposition='id'))), device_properties=dict(type='dict', disposition='/device_properties', options=dict(device_vendor=dict(type='str', disposition='device_vendor'), device_model=dict(type='str', disposition='device_model'), link_speed_in_mbps=dict(type='int', disposition='link_speed_in_mbps'))), ip_address=dict(type='str', disposition='/ip_address'), site_key=dict(type='str', no_log=True, disposition='/site_key'), address_space=dict(type='dict', disposition='/address_space', options=dict(address_prefixes=dict(type='list', disposition='address_prefixes', elements='str'))), bgp_properties=dict(type='dict', disposition='/bgp_properties', options=dict(asn=dict(type='int', disposition='asn'), bgp_peering_address=dict(type='str', disposition='bgp_peering_address'), peer_weight=dict(type='int', disposition='peer_weight'), bgp_peering_addresses=dict(type='list', disposition='bgp_peering_addresses', elements='dict', options=dict(ipconfiguration_id=dict(type='str', disposition='ipconfiguration_id'), default_bgp_ip_addresses=dict(type='list', updatable=False, disposition='default_bgp_ip_addresses', elements='str'), custom_bgp_ip_addresses=dict(type='list', disposition='custom_bgp_ip_addresses', elements='str'), tunnel_ip_addresses=dict(type='list', updatable=False, disposition='tunnel_ip_addresses', elements='str'))))), is_security_site=dict(type='bool', disposition='/is_security_site'), vpn_site_links=dict(type='list', disposition='/vpn_site_links', elements='dict', options=dict(name=dict(type='str', disposition='name'), link_properties=dict(type='dict', disposition='link_properties', options=dict(link_provider_name=dict(type='str', disposition='link_provider_name'), link_speed_in_mbps=dict(type='int', disposition='link_speed_in_mbps'))), ip_address=dict(type='str', disposition='ip_address'), fqdn=dict(type='str', disposition='fqdn'), bgp_properties=dict(type='dict', disposition='bgp_properties', options=dict(asn=dict(type='int', disposition='asn'), bgp_peering_address=dict(type='str', disposition='bgp_peering_address'))))), o365_policy=dict(type='dict', disposition='/o365_policy', options=dict(break_out_categories=dict(type='dict', disposition='break_out_categories', options=dict(allow=dict(type='bool', disposition='allow'), optimize=dict(type='bool', disposition='optimize'), default=dict(type='bool', disposition='default'))))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.location = None
        self.body = {}
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMVpnSite, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

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
            response = self.network_client.vpn_sites.begin_create_or_update(resource_group_name=self.resource_group, vpn_site_name=self.name, vpn_site_parameters=self.body)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the VpnSite instance.')
            self.fail('Error creating the VpnSite instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_resource(self):
        try:
            response = self.network_client.vpn_sites.begin_delete(resource_group_name=self.resource_group, vpn_site_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the VpnSite instance.')
            self.fail('Error deleting the VpnSite instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        try:
            response = self.network_client.vpn_sites.get(resource_group_name=self.resource_group, vpn_site_name=self.name)
        except ResourceNotFoundError as e:
            return False
        return response.as_dict()