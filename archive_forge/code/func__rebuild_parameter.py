import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_parameter(data, requires_grad, backward_hooks):
    param = torch.nn.Parameter(data, requires_grad)
    param._backward_hooks = backward_hooks
    return param