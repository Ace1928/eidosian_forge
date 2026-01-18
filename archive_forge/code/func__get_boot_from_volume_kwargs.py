import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def _get_boot_from_volume_kwargs(self, image, boot_from_volume, boot_volume, volume_size, terminate_volume, volumes, kwargs):
    """Return block device mappings

        :param image: Image dict, name or id to boot with.

        """
    if boot_volume or boot_from_volume or volumes:
        kwargs.setdefault('block_device_mapping_v2', [])
    else:
        return kwargs
    if boot_volume:
        volume = self.get_volume(boot_volume)
        if not volume:
            raise exceptions.SDKException('Volume {boot_volume} is not a valid volume in {cloud}:{region}'.format(boot_volume=boot_volume, cloud=self.name, region=self._compute_region))
        block_mapping = {'boot_index': '0', 'delete_on_termination': terminate_volume, 'destination_type': 'volume', 'uuid': volume['id'], 'source_type': 'volume'}
        kwargs['block_device_mapping_v2'].append(block_mapping)
        kwargs['imageRef'] = ''
    elif boot_from_volume:
        if isinstance(image, dict):
            image_obj = image
        else:
            image_obj = self.get_image(image)
        if not image_obj:
            raise exceptions.SDKException('Image {image} is not a valid image in {cloud}:{region}'.format(image=image, cloud=self.name, region=self._compute_region))
        block_mapping = {'boot_index': '0', 'delete_on_termination': terminate_volume, 'destination_type': 'volume', 'uuid': image_obj['id'], 'source_type': 'image', 'volume_size': volume_size}
        kwargs['imageRef'] = ''
        kwargs['block_device_mapping_v2'].append(block_mapping)
    if volumes and kwargs['imageRef']:
        block_mapping = {u'boot_index': 0, u'delete_on_termination': True, u'destination_type': u'local', u'source_type': u'image', u'uuid': kwargs['imageRef']}
        kwargs['block_device_mapping_v2'].append(block_mapping)
    for volume in volumes:
        volume_obj = self.get_volume(volume)
        if not volume_obj:
            raise exceptions.SDKException('Volume {volume} is not a valid volume in {cloud}:{region}'.format(volume=volume, cloud=self.name, region=self._compute_region))
        block_mapping = {'boot_index': '-1', 'delete_on_termination': False, 'destination_type': 'volume', 'uuid': volume_obj['id'], 'source_type': 'volume'}
        kwargs['block_device_mapping_v2'].append(block_mapping)
    return kwargs