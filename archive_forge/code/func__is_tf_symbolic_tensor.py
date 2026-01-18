import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
def _is_tf_symbolic_tensor(x):
    import tensorflow as tf
    if hasattr(tf, 'is_symbolic_tensor'):
        return tf.is_symbolic_tensor(x)
    return type(x) == tf.Tensor