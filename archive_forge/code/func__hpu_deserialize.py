import difflib
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from enum import Enum
from ._utils import _import_dotted_name
from torch._sources import get_source_lines_and_file
from torch.types import Storage
from torch.storage import _get_dtype_from_pickle_storage_type
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO
from typing_extensions import TypeAlias  # Python 3.10+
import copyreg
import pickle
import pathlib
import torch._weights_only_unpickler as _weights_only_unpickler
def _hpu_deserialize(obj, location):
    if location.startswith('hpu'):
        hpu = getattr(torch, 'hpu', None)
        assert hpu is not None, 'HPU device module is not loaded'
        device = validate_hpu_device(location)
        if getattr(obj, '_torch_load_uninitialized', False):
            with hpu.device(device):
                return torch.UntypedStorage(obj.nbytes(), device=torch.device(location))
        else:
            return obj.hpu(device)