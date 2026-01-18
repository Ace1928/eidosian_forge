import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
def _stride_or_default(stride: Optional['torch._prims_common.StrideType'], *, shape: 'torch._prims_common.ShapeType') -> 'torch._prims_common.StrideType':
    return stride if stride is not None else utils.make_contiguous_strides_for(shape)