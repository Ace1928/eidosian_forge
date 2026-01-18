import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm
from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
def _asdict_inner(obj):
    if _is_dataclass_instance(obj):
        result = {}
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name))
            if not f.init or value != f.default or f.metadata.get('include_in_asdict_even_if_is_default', False):
                result[f.name] = value
        return result
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[_asdict_inner(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        return type(obj)((_asdict_inner(v) for v in obj))
    elif isinstance(obj, dict):
        return {_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()}
    else:
        return copy.deepcopy(obj)