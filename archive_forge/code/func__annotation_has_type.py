import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _annotation_has_type(*, typ, annotation):
    if hasattr(annotation, '__args__'):
        for a in annotation.__args__:
            if _annotation_has_type(typ=typ, annotation=a):
                return True
        return False
    return typ is annotation