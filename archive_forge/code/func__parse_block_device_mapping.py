import abc
import contextlib
import copy
import hashlib
import os
import threading
from oslo_utils import reflection
from oslo_utils import strutils
import requests
from novaclient import exceptions
from novaclient import utils
def _parse_block_device_mapping(self, block_device_mapping):
    """Parses legacy block device mapping."""
    bdm = []
    for device_name, mapping in block_device_mapping.items():
        bdm_dict = {'device_name': device_name}
        mapping_parts = mapping.split(':')
        source_id = mapping_parts[0]
        if len(mapping_parts) == 1:
            bdm_dict['volume_id'] = source_id
        elif len(mapping_parts) > 1:
            source_type = mapping_parts[1]
            if source_type.startswith('snap'):
                bdm_dict['snapshot_id'] = source_id
            else:
                bdm_dict['volume_id'] = source_id
        if len(mapping_parts) > 2 and mapping_parts[2]:
            bdm_dict['volume_size'] = str(int(mapping_parts[2]))
        if len(mapping_parts) > 3:
            bdm_dict['delete_on_termination'] = mapping_parts[3]
        bdm.append(bdm_dict)
    return bdm