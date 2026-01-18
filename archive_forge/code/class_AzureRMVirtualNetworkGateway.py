from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMVirtualNetworkGateway(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), ip_configurations=dict(type='list', default=None, elements='dict', options=ip_configuration_spec), gateway_type=dict(type='str', default='vpn', choices=['vpn', 'express_route']), vpn_type=dict(type='str', default='route_based', choices=['route_based', 'policy_based']), vpn_gateway_generation=dict(type='str', default='Generation1', choices=['None', 'Generation1', 'Generation2']), enable_bgp=dict(type='bool', default=False), sku=dict(type='str', default='VpnGw1', choices=['VpnGw1', 'VpnGw2', 'VpnGw3', 'Standard', 'Basic', 'HighPerformance']), bgp_settings=dict(type='dict', options=bgp_spec), virtual_network=dict(type='raw', aliases=['virtual_network_name']))
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.ip_configurations = None
        self.gateway_type = None
        self.vpn_type = None
        self.enable_bgp = None
        self.sku = None
        self.vpn_gateway_generation = None
        self.bgp_settings = None
        required_if = [('state', 'present', ['virtual_network'])]
        self.results = dict(changed=False, state=dict())
        super(AzureRMVirtualNetworkGateway, self).__init__(derived_arg_spec=self.module_arg_spec, required_if=required_if, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        vgw = None
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.virtual_network = self.parse_resource_to_dict(self.virtual_network)
        resource_group = self.get_resource_group(self.resource_group)
        try:
            vgw = self.network_client.virtual_network_gateways.get(self.resource_group, self.name)
            if self.state == 'absent':
                self.log("CHANGED: vnet exists but requested state is 'absent'")
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                self.log("CHANGED: VPN Gateway {0} does not exist but requested state is 'present'".format(self.name))
                changed = True
        if vgw:
            results = vgw_to_dict(vgw)
            if self.state == 'present':
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                sku = dict(name=self.sku, tier=self.sku)
                if sku != results['sku']:
                    changed = True
                if self.enable_bgp != results['enable_bgp']:
                    changed = True
                if self.bgp_settings and self.bgp_settings['asn'] != results['bgp_settings']['asn']:
                    changed = True
        self.results['changed'] = changed
        self.results['id'] = results.get('id')
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                if not self.sku:
                    self.fail('Parameter error: sku is required when creating a vpn gateway')
                if not self.ip_configurations:
                    self.fail('Parameter error: ip_configurations required when creating a vpn gateway')
                subnet = self.network_models.SubResource(id='/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks/{2}/subnets/GatewaySubnet'.format(self.virtual_network['subscription_id'], self.virtual_network['resource_group'], self.virtual_network['name']))
                public_ip_address = self.network_models.SubResource(id='/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/publicIPAddresses/{2}'.format(self.virtual_network['subscription_id'], self.virtual_network['resource_group'], self.ip_configurations[0]['public_ip_address_name']))
                vgw_ip_configurations = [self.network_models.VirtualNetworkGatewayIPConfiguration(private_ip_allocation_method=ip_config.get('private_ip_allocation_method'), subnet=subnet, public_ip_address=public_ip_address, name='default') for ip_config in self.ip_configurations]
                vgw_sku = self.network_models.VirtualNetworkGatewaySku(name=self.sku, tier=self.sku)
                vgw_bgp_settings = self.network_models.BgpSettings(asn=self.bgp_settings.get('asn')) if self.bgp_settings else None
                vgw = self.network_models.VirtualNetworkGateway(location=self.location, ip_configurations=vgw_ip_configurations, gateway_type=_snake_to_camel(self.gateway_type, True), vpn_type=_snake_to_camel(self.vpn_type, True), vpn_gateway_generation=_snake_to_camel(self.vpn_gateway_generation, True), enable_bgp=self.enable_bgp, sku=vgw_sku, bgp_settings=vgw_bgp_settings)
                if self.tags:
                    vgw.tags = self.tags
                results = self.create_or_update_vgw(vgw)
            else:
                results = self.delete_vgw()
        if self.state == 'present':
            self.results['id'] = results.get('id')
        return self.results

    def create_or_update_vgw(self, vgw):
        try:
            poller = self.network_client.virtual_network_gateways.begin_create_or_update(self.resource_group, self.name, vgw)
            new_vgw = self.get_poller_result(poller)
            return vgw_to_dict(new_vgw)
        except Exception as exc:
            self.fail('Error creating or updating virtual network gateway {0} - {1}'.format(self.name, str(exc)))

    def delete_vgw(self):
        try:
            poller = self.network_client.virtual_network_gateways.begin_delete(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting virtual network gateway {0} - {1}'.format(self.name, str(exc)))
        return True