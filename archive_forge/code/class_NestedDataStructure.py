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
class NestedDataStructure:

    def __init__(self, data=None):
        self.data = data if data is not None else []

    def flatten(self, data=None):
        data = data if data is not None else self.data
        if isinstance(data, dict):
            return self.flatten(list(data.values()))
        elif isinstance(data, (list, tuple)):
            return [flattened for item in data for flattened in self.flatten(item)]
        else:
            return [data]