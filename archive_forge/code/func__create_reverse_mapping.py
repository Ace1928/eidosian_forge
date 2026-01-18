import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _create_reverse_mapping(my_name, all_names, all_device_maps):
    reverse_device_maps: Dict[str, Dict[torch.device, torch.device]] = {}
    for node in all_names:
        if my_name in all_device_maps[node]:
            reverse_device_maps[node] = {v: k for k, v in all_device_maps[node][my_name].items()}
    return reverse_device_maps