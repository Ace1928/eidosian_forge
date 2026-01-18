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
def _build_block_device_mapping(cls, bdm):
    if not bdm:
        return None
    bdm_dict = {}
    for mapping in bdm:
        mapping_parts = []
        snapshot_id = mapping.get(cls.BLOCK_DEVICE_MAPPING_SNAPSHOT_ID)
        if snapshot_id:
            mapping_parts.append(snapshot_id)
            mapping_parts.append('snap')
        else:
            volume_id = mapping.get(cls.BLOCK_DEVICE_MAPPING_VOLUME_ID)
            mapping_parts.append(volume_id)
            mapping_parts.append('')
        volume_size = mapping.get(cls.BLOCK_DEVICE_MAPPING_VOLUME_SIZE)
        delete = mapping.get(cls.BLOCK_DEVICE_MAPPING_DELETE_ON_TERM)
        if volume_size:
            mapping_parts.append(str(volume_size))
        else:
            mapping_parts.append('')
        if delete:
            mapping_parts.append(str(delete))
        device_name = mapping.get(cls.BLOCK_DEVICE_MAPPING_DEVICE_NAME)
        bdm_dict[device_name] = ':'.join(mapping_parts)
    return bdm_dict