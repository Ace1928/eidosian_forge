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
def _open_file_like(name_or_buffer, mode):
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)
    elif 'w' in mode:
        return _open_buffer_writer(name_or_buffer)
    elif 'r' in mode:
        return _open_buffer_reader(name_or_buffer)
    else:
        raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")