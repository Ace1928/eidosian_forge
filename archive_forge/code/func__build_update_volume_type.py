from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_update_volume_type(self, volume_type):
    update = {}
    allowed_attributes = ['is_public', 'description', 'name']
    type_attributes = {k: self.params[k] for k in allowed_attributes if k in self.params and self.params.get(k) is not None and (self.params.get(k) != volume_type.get(k))}
    if type_attributes:
        update['type_attributes'] = type_attributes
    return update