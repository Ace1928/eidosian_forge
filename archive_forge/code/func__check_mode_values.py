from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _check_mode_values(self, state, server, meta):
    """Builds return values for check mode"""
    changed = False
    if state == 'present':
        update = self._build_update(server.metadata, meta)
        if update:
            changed = True
            new_metadata = server.metadata or {}
            new_metadata.update(update)
            server.metadata = new_metadata
    else:
        keys = self._get_keys_to_delete(server.metadata, meta)
        for k in keys:
            server.meta.pop(k)
    return dict(changed=changed, server=server.to_dict(computed=False))