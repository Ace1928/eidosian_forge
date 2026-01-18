import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices, is_static_group=True):
    for node in all_names:
        devices = all_devices[node]
        if len(set(devices)) != len(devices):
            raise ValueError(f'Node {node} has duplicated devices\ndevices = {devices}')
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(f'Node {node} has devices with invalid indices\ndevices = {devices}\ndevice count = {all_device_counts[node]}')
    for source_node in all_names:
        if is_static_group and (not set(all_device_maps[source_node].keys()).issubset(all_names)):
            raise ValueError(f'Node {source_node} has invalid target node names in its device maps\ndevice maps = {all_device_maps[source_node].keys()}\nnode names = {all_names}')
        for target_node, map_ in all_device_maps[source_node].items():
            if len(set(map_.values())) != len(map_):
                raise ValueError(f'Node {source_node} has duplicated target devices in its device map for {target_node}\ndevice map = {map_}')
            if all_devices[source_node]:
                if not set(map_.keys()).issubset(all_devices[source_node]):
                    raise ValueError(f'Node {source_node} has unexpected source devices in its device map for {target_node}\ndevice map = {map_}\ndevices = {all_devices[source_node]}')
            elif not _tensorpipe_validate_devices(map_.keys(), all_device_counts[source_node]):
                raise ValueError(f'Node {source_node} has source devices with invalid indices in its device map for {target_node}\ndevice map = {map_}\ndevice count = {all_device_counts[source_node]}')
            if all_devices.get(target_node, []):
                if not set(map_.values()).issubset(all_devices[target_node]):
                    raise ValueError(f'Node {source_node} has unexpected target devices in its device map for {target_node}\ndevice map = {map_}\ndevices = {all_devices[target_node]}')
            elif target_node in all_device_counts and (not _tensorpipe_validate_devices(map_.values(), all_device_counts[target_node])):
                raise ValueError(f'Node {source_node} has target devices with invalid indices in its device map for {target_node}\ndevice map = {map_}\ndevice count = {all_device_counts[target_node]}')