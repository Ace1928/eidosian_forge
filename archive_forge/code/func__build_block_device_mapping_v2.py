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
@classmethod
def _build_block_device_mapping_v2(cls, bdm_v2):
    if not bdm_v2:
        return None
    bdm_v2_list = []
    for mapping in bdm_v2:
        bmd_dict = None
        if mapping.get(cls.BLOCK_DEVICE_MAPPING_VOLUME_ID):
            bmd_dict = {'uuid': mapping.get(cls.BLOCK_DEVICE_MAPPING_VOLUME_ID), 'source_type': 'volume', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}
        elif mapping.get(cls.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID):
            bmd_dict = {'uuid': mapping.get(cls.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID), 'source_type': 'snapshot', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}
        elif mapping.get(cls.BLOCK_DEVICE_MAPPING_IMAGE):
            bmd_dict = {'uuid': mapping.get(cls.BLOCK_DEVICE_MAPPING_IMAGE), 'source_type': 'image', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}
        elif mapping.get(cls.BLOCK_DEVICE_MAPPING_SWAP_SIZE):
            bmd_dict = {'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'delete_on_termination': True, 'guest_format': 'swap', 'volume_size': mapping.get(cls.BLOCK_DEVICE_MAPPING_SWAP_SIZE)}
        elif mapping.get(cls.BLOCK_DEVICE_MAPPING_EPHEMERAL_SIZE) or mapping.get(cls.BLOCK_DEVICE_MAPPING_EPHEMERAL_FORMAT):
            bmd_dict = {'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'delete_on_termination': True}
            ephemeral_size = mapping.get(cls.BLOCK_DEVICE_MAPPING_EPHEMERAL_SIZE)
            if ephemeral_size:
                bmd_dict.update({'volume_size': ephemeral_size})
            ephemeral_format = mapping.get(cls.BLOCK_DEVICE_MAPPING_EPHEMERAL_FORMAT)
            if ephemeral_format:
                bmd_dict.update({'guest_format': ephemeral_format})
        device_name = mapping.get(cls.BLOCK_DEVICE_MAPPING_DEVICE_NAME)
        if device_name:
            bmd_dict[cls.BLOCK_DEVICE_MAPPING_DEVICE_NAME] = device_name
        update_props = (cls.BLOCK_DEVICE_MAPPING_DEVICE_TYPE, cls.BLOCK_DEVICE_MAPPING_DISK_BUS, cls.BLOCK_DEVICE_MAPPING_BOOT_INDEX, cls.BLOCK_DEVICE_MAPPING_VOLUME_SIZE, cls.BLOCK_DEVICE_MAPPING_DELETE_ON_TERM)
        for update_prop in update_props:
            if mapping.get(update_prop) is not None:
                bmd_dict[update_prop] = mapping.get(update_prop)
        if bmd_dict:
            bdm_v2_list.append(bmd_dict)
    return bdm_v2_list