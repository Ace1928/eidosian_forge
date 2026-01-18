from typing import Any, Callable, List, no_type_check
import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass
def _set_params_to_optimize(self, params):
    if params is not None:
        self.params_to_optimize = set(params)