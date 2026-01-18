from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _validate_update(self, subnet, update):
    """ Check for differences in non-updatable values """
    for attr in ('cidr', 'ip_version', 'ipv6_ra_mode', 'ipv6_address_mode', 'prefix_length', 'use_default_subnet_pool'):
        if attr in update and update[attr] != subnet[attr]:
            self.fail_json(msg='Cannot update {0} in existing subnet'.format(attr))