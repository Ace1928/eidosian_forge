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
def _privateuse1_tag(obj):
    backend_name = torch._C._get_privateuse1_backend_name()
    if obj.device.type == backend_name:
        if obj.device.index is None:
            return backend_name
        else:
            return backend_name + ':' + str(obj.device.index)