import collections
import enum
from typing import cast, Dict, List, Set, Tuple
import torch
import torch.distributed as dist
from ._utils import _group_membership_management, _update_group_membership
from . import api
from . import constants as rpc_constants
def _tensorpipe_construct_rpc_backend_options_handler(rpc_timeout, init_method, num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS, _transports=None, _channels=None, **kwargs):
    from . import TensorPipeRpcBackendOptions
    return TensorPipeRpcBackendOptions(rpc_timeout=rpc_timeout, init_method=init_method, num_worker_threads=num_worker_threads, _transports=_transports, _channels=_channels)