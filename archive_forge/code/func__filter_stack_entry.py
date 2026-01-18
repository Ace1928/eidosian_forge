import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _filter_stack_entry(entry):
    filtered_entries = [('autograd/__init__', '_make_grads'), ('autograd/__init__', 'backward'), ('torch/tensor', 'backward'), ('_internal/common_utils', 'prof_callable'), ('_internal/common_utils', 'prof_func_call'), ('_internal/common_utils', 'prof_meth_call')]
    return all((not (f[0] in entry and f[1] in entry) for f in filtered_entries))