import functools
import logging
import math
import os
import warnings
from contextlib import ExitStack
from functools import partial
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import init
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self, override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.types import _DEVICE
class _Int8LinearInference(_Linear8bitLt):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, has_fp16_weights=False, **kwargs)