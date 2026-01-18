import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _tensorpipe_validate_devices(devices, device_count):
    return all((d.type == 'cpu' or (d.type == 'cuda' and 0 <= d.index < device_count) for d in devices))