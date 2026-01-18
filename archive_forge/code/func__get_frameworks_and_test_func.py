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
def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {'pt': is_torch_tensor, 'tf': is_tf_tensor, 'jax': is_jax_tensor, 'np': is_numpy_array}
    preferred_framework = infer_framework_from_repr(x)
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != 'np':
        frameworks.append('np')
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, 'np']])
    return {f: framework_to_test[f] for f in frameworks}