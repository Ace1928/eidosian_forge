from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
class AzureRMDNSZone(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(choices=['present', 'absent'], default='present', type='str'), type=dict(type='str', choices=['private', 'public']), registration_virtual_networks=dict(type='list', elements='raw'), resolution_virtual_networks=dict(type='list', elements='raw'))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.tags = None
        self.type = None
        self.registration_virtual_networks = None
        self.resolution_virtual_networks = None
        super(AzureRMDNSZone, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        zone = None
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.registration_virtual_networks = self.preprocess_vn_list(self.registration_virtual_networks)
        self.resolution_virtual_networks = self.preprocess_vn_list(self.resolution_virtual_networks)
        self.results['check_mode'] = self.check_mode
        self.get_resource_group(self.resource_group)
        changed = False
        results = dict()
        try:
            self.log('Fetching DNS zone {0}'.format(self.name))
            zone = self.dns_client.zones.get(self.resource_group, self.name)
            results = zone_to_dict(zone)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                if self.type and results['type'] != self.type:
                    changed = True
                    results['type'] = self.type
                if self.resolution_virtual_networks:
                    if set(self.resolution_virtual_networks) != set(results['resolution_virtual_networks'] or []):
                        changed = True
                        results['resolution_virtual_networks'] = self.resolution_virtual_networks
                else:
                    self.resolution_virtual_networks = results['resolution_virtual_networks']
                if self.registration_virtual_networks:
                    if set(self.registration_virtual_networks) != set(results['registration_virtual_networks'] or []):
                        changed = True
                        results['registration_virtual_networks'] = self.registration_virtual_networks
                else:
                    self.registration_virtual_networks = results['registration_virtual_networks']
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                zone = self.dns_models.Zone(zone_type=str.capitalize(self.type) if self.type else None, tags=self.tags, location='global')
                if self.resolution_virtual_networks:
                    zone.resolution_virtual_networks = self.construct_subresource_list(self.resolution_virtual_networks)
                if self.registration_virtual_networks:
                    zone.registration_virtual_networks = self.construct_subresource_list(self.registration_virtual_networks)
                self.results['state'] = self.create_or_update_zone(zone)
            elif self.state == 'absent':
                self.delete_zone()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_zone(self, zone):
        try:
            new_zone = self.dns_client.zones.create_or_update(self.resource_group, self.name, zone)
        except Exception as exc:
            self.fail('Error creating or updating zone {0} - {1}'.format(self.name, exc.message or str(exc)))
        return zone_to_dict(new_zone)

    def delete_zone(self):
        try:
            poller = self.dns_client.zones.begin_delete(self.resource_group, self.name)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting zone {0} - {1}'.format(self.name, exc.message or str(exc)))
        return result

    def preprocess_vn_list(self, vn_list):
        return [self.parse_vn_id(x) for x in vn_list] if vn_list else None

    def parse_vn_id(self, vn):
        vn_dict = self.parse_resource_to_dict(vn) if not isinstance(vn, dict) else vn
        return format_resource_id(val=vn_dict['name'], subscription_id=vn_dict.get('subscription') or self.subscription_id, namespace='Microsoft.Network', types='virtualNetworks', resource_group=vn_dict.get('resource_group') or self.resource_group)

    def construct_subresource_list(self, raw):
        return [self.dns_models.SubResource(id=x) for x in raw] if raw else None