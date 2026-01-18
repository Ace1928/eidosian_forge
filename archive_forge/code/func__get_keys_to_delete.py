from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _get_keys_to_delete(self, current=None, requested=None):
    current = current or {}
    requested = requested or {}
    return set(current.keys() & requested.keys())