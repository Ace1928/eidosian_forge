import functools
import logging
import time
from typing import Any, Dict, List, Tuple
import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
def _time_logger(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time_ns()
        func_return = func(*args, **kwargs)
        time_spent = time.time_ns() - t1
        msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
        msg_dict['time_spent'] = f'{time_spent}ns'
        _c10d_logger.debug(msg_dict)
        return func_return
    return wrapper