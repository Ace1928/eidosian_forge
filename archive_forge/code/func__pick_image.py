import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def _pick_image(self):
    """Pick a sensible image to run tests with.

        This returns None if the image service is not present.
        """
    if not self.user_cloud.has_service('image'):
        return None
    images = self.user_cloud.list_images()
    image_name = os.environ.get('OPENSTACKSDK_IMAGE')
    if not image_name:
        image_name = _get_resource_value('image_name')
    if image_name:
        for image in images:
            if image.name == image_name:
                return image
        raise self.failureException("Cloud does not have image '%s'", image_name)
    for image in images:
        if image.name.startswith('cirros') and image.name.endswith('-uec'):
            return image
    for image in images:
        if image.name.startswith('cirros') and image.disk_format == 'qcow2':
            return image
    for image in images:
        if image.name.lower().startswith('ubuntu'):
            return image
    for image in images:
        if image.name.lower().startswith('centos'):
            return image
    raise self.failureException('No sensible image found')