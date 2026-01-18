import collections
from functools import wraps
from typing import Callable, DefaultDict, Dict, List
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._subclasses.meta_utils import safe_is_leaf
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .functional_utils import (
from .schemas import (
from .subclass_utils import create_subclass_meta
from .utils import _get_autocast_states, KNOWN_TYPES, strict_zip
from a multi-output view call"

This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the analysis here constructs view and mutation metadata from running
a functionalized version of the graph under compilation.
