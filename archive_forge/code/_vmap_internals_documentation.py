import functools
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten

    Please use torch.vmap instead of this API.
    