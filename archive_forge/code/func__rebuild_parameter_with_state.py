import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    param = torch.nn.Parameter(data, requires_grad)
    param._backward_hooks = backward_hooks
    param = _set_obj_state(param, state)
    return param