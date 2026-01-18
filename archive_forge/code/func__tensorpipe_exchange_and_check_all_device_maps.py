import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _tensorpipe_exchange_and_check_all_device_maps(my_name, my_device_count, my_device_maps, my_devices, group):
    gathered: List[Tuple[str, int, Dict[str, Dict[torch.device, torch.device]], List[torch.device]]] = [('', 0, {}, []) for _ in range(group.size())]
    dist.all_gather_object(gathered, (my_name, my_device_count, my_device_maps, my_devices), group)
    all_names = [name for name, _, _, _ in gathered]
    all_device_counts = {name: count for name, count, _, _ in gathered}
    all_device_maps = {name: map_ for name, _, map_, _ in gathered}
    all_devices = {name: devices for name, _, _, devices in gathered}
    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    my_devices = _create_device_list(my_devices, my_device_maps, reverse_device_maps)
    return (reverse_device_maps, my_devices)