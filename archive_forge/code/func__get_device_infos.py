import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _get_device_infos():
    from . import TensorPipeAgent
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    opts = agent._get_backend_options()
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available() and opts.devices:
        torch.cuda.init()
    return (device_count, opts.device_maps, opts.devices)