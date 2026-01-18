from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _resolve_visibility(self):
    """resolve a visibility value to be compatible with older versions"""
    if self.params['visibility']:
        return self.params['visibility']
    if self.params['is_public'] is not None:
        return 'public' if self.params['is_public'] else 'private'
    return None