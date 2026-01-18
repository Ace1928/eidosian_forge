import functools
import logging
import time
from typing import Any, Dict, List, Tuple
import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
def _get_msg_dict(func_name, *args, **kwargs) -> Dict[str, Any]:
    if dist.is_initialized():
        msg_dict = {'func_name': f'{func_name}', 'args': f'{args}, {kwargs}', 'pg_name': f'{dist._get_process_group_name(kwargs.get('pg'))}', 'backend': f'{dist.get_backend(kwargs.get('group'))}', 'world_size': f'{dist.get_world_size()}', 'group_size': f'{dist.get_world_size(kwargs.get('group'))}', 'global_rank': f'{dist.get_rank()}', 'local_rank': f'{dist.get_rank(kwargs.get('group'))}'}
        if msg_dict['backend'] == 'nccl':
            nccl_version = torch.cuda.nccl.version()
            msg_dict['nccl_version'] = '.'.join((str(v) for v in nccl_version))
    else:
        msg_dict = {'func_name': f'{func_name}', 'args': f'{args}, {kwargs}'}
    return msg_dict