import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Tuple, Union
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.types import _PATH, _Stateful
def _materialize_tensors(collection: Any) -> Any:

    def _load_tensor(t: _NotYetLoadedTensor) -> Tensor:
        return t._load_tensor()
    return apply_to_collection(collection, dtype=_NotYetLoadedTensor, function=_load_tensor)