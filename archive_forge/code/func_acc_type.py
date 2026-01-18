import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
def acc_type(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 'tl.float32'
    return f'tl.{dtype}'.replace('torch.', '')