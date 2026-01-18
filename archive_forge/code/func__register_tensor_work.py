import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _register_tensor_work(tensor_or_list, work_or_list):
    if not isinstance(tensor_or_list, list):
        tensor_or_list = [tensor_or_list]
    if not isinstance(work_or_list, list):
        reg = _WaitRegistration(work_or_list)
        for tensor in tensor_or_list:
            reg._register_tensor_ptr(tensor.data_ptr())
    else:
        for tensor, work in zip(tensor_or_list, work_or_list):
            reg = _WaitRegistration(work)
            reg._register_tensor_ptr(tensor.data_ptr())