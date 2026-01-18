import copy
import ipaddress
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import port as neutron_port
from heat.engine.resources.openstack.neutron import subnet
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import server_base
from heat.engine import support
from heat.engine import translation
from heat.rpc import api as rpc_api
def _validate_block_device_mapping(self):
    bdm = self.properties[self.BLOCK_DEVICE_MAPPING] or []
    bdm_v2 = self.properties[self.BLOCK_DEVICE_MAPPING_V2] or []
    image = self.properties[self.IMAGE]
    if bdm and bdm_v2:
        raise exception.ResourcePropertyConflict(self.BLOCK_DEVICE_MAPPING, self.BLOCK_DEVICE_MAPPING_V2)
    bootable = image is not None
    for mapping in bdm:
        device_name = mapping[self.BLOCK_DEVICE_MAPPING_DEVICE_NAME]
        if device_name == 'vda':
            bootable = True
        volume_id = mapping.get(self.BLOCK_DEVICE_MAPPING_VOLUME_ID)
        snapshot_id = mapping.get(self.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID)
        if volume_id is not None and snapshot_id is not None:
            raise exception.ResourcePropertyConflict(self.BLOCK_DEVICE_MAPPING_VOLUME_ID, self.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID)
        if volume_id is None and snapshot_id is None:
            msg = _('Either volume_id or snapshot_id must be specified for device mapping %s') % device_name
            raise exception.StackValidationFailed(message=msg)
    bootable_devs = [image]
    for mapping in bdm_v2:
        volume_id = mapping.get(self.BLOCK_DEVICE_MAPPING_VOLUME_ID)
        snapshot_id = mapping.get(self.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID)
        image_id = mapping.get(self.BLOCK_DEVICE_MAPPING_IMAGE)
        boot_index = mapping.get(self.BLOCK_DEVICE_MAPPING_BOOT_INDEX)
        swap_size = mapping.get(self.BLOCK_DEVICE_MAPPING_SWAP_SIZE)
        ephemeral = mapping.get(self.BLOCK_DEVICE_MAPPING_EPHEMERAL_SIZE) or mapping.get(self.BLOCK_DEVICE_MAPPING_EPHEMERAL_FORMAT)
        property_tuple = (volume_id, snapshot_id, image_id, swap_size, ephemeral)
        if property_tuple.count(None) < 4:
            raise exception.ResourcePropertyConflict(self.BLOCK_DEVICE_MAPPING_VOLUME_ID, self.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID, self.BLOCK_DEVICE_MAPPING_IMAGE, self.BLOCK_DEVICE_MAPPING_SWAP_SIZE, self.BLOCK_DEVICE_MAPPING_EPHEMERAL_SIZE, self.BLOCK_DEVICE_MAPPING_EPHEMERAL_FORMAT)
        if property_tuple.count(None) == 5:
            msg = _('Either volume_id, snapshot_id, image_id, swap_size, ephemeral_size or ephemeral_format must be specified.')
            raise exception.StackValidationFailed(message=msg)
        if any((volume_id is not None, snapshot_id is not None, image_id is not None)):
            if boot_index is None or boot_index == 0:
                bootable = True
                bootable_devs.append(volume_id)
                bootable_devs.append(snapshot_id)
                bootable_devs.append(image_id)
    if not bootable:
        msg = _('Neither image nor bootable volume is specified for instance %s') % self.name
        raise exception.StackValidationFailed(message=msg)
    if bdm_v2 and len(list((dev for dev in bootable_devs if dev is not None))) != 1:
        msg = _('Multiple bootable sources for instance %s.') % self.name
        raise exception.StackValidationFailed(message=msg)