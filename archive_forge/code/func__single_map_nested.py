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
def _single_map_nested(args):
    """Apply a function recursively to each element of a nested data struct."""
    function, data_struct, types, rank, disable_tqdm, desc = args
    if not isinstance(data_struct, dict) and (not isinstance(data_struct, types)):
        return function(data_struct)
    if rank is not None and logging.get_verbosity() < logging.WARNING:
        logging.set_verbosity_warning()
    if rank is not None and (not disable_tqdm) and any(('notebook' in tqdm_cls.__name__ for tqdm_cls in tqdm.__mro__)):
        print(' ', end='', flush=True)
    pbar_iterable = data_struct.items() if isinstance(data_struct, dict) else data_struct
    pbar_desc = (desc + ' ' if desc is not None else '') + '#' + str(rank) if rank is not None else desc
    with hf_tqdm(pbar_iterable, disable=disable_tqdm, position=rank, unit='obj', desc=pbar_desc) as pbar:
        if isinstance(data_struct, dict):
            return {k: _single_map_nested((function, v, types, None, True, None)) for k, v in pbar}
        else:
            mapped = [_single_map_nested((function, v, types, None, True, None)) for v in pbar]
            if isinstance(data_struct, list):
                return mapped
            elif isinstance(data_struct, tuple):
                return tuple(mapped)
            else:
                return np.array(mapped)