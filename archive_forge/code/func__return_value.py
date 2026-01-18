from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _return_value(self, image_name_or_id):
    image = self.conn.image.find_image(image_name_or_id)
    if image:
        image = image.to_dict(computed=False)
    return image