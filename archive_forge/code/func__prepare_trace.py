from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
def _prepare_trace(self):
    self.entered = True
    _prepare_profiler(self.config(), self.kineto_activities)