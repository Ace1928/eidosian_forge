import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _check_randomness_arg(randomness):
    if randomness not in ['error', 'different', 'same']:
        raise RuntimeError(f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}")