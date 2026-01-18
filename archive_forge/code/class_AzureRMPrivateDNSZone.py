from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMPrivateDNSZone(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.tags = None
        super(AzureRMPrivateDNSZone, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        zone = None
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.results['check_mode'] = self.check_mode
        self.get_resource_group(self.resource_group)
        changed = False
        results = dict()
        try:
            self.log('Fetching private DNS zone {0}'.format(self.name))
            zone = self.private_dns_client.private_zones.get(self.resource_group, self.name)
            results = zone_to_dict(zone)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
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
                zone = self.private_dns_models.PrivateZone(tags=self.tags, location='global')
                self.results['state'] = self.create_or_update_zone(zone)
            elif self.state == 'absent':
                self.delete_zone()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_zone(self, zone):
        try:
            new_zone = self.private_dns_client.private_zones.begin_create_or_update(self.resource_group, self.name, zone)
            if isinstance(new_zone, LROPoller):
                new_zone = self.get_poller_result(new_zone)
        except Exception as exc:
            self.fail('Error creating or updating zone {0} - {1}'.format(self.name, exc.message or str(exc)))
        return zone_to_dict(new_zone)

    def delete_zone(self):
        try:
            poller = self.private_dns_client.private_zones.begin_delete(self.resource_group, self.name)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting zone {0} - {1}'.format(self.name, exc.message or str(exc)))
        return result