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
def _validate_privateuse1_device(location, backend_name):
    """
    Check whether the device index of privateuse1 is valid

    Register a device_module of privateuse1 by torch._register_device_module.
    Implement the following methods in device_module like cuda:
    device_module._utils._get_device_index(location, True),
    device_module.device_count().

    Args:
        location: string of device
        backend_name: the name of privateuse1, which can be renamed

    Returns:
        device_index: int
    """
    if not hasattr(torch, backend_name):
        raise RuntimeError(f"The {backend_name.upper()} device module is not registered. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.")
    device_module = getattr(torch, backend_name)
    if hasattr(device_module, '_utils') and hasattr(device_module._utils, '_get_device_index'):
        device_index = device_module._utils._get_device_index(location, True)
    else:
        device = torch.device(location)
        device_index = device.index if device.index else 0
    if hasattr(device_module, 'is_available') and (not device_module.is_available()):
        raise RuntimeError(f"Attempting to deserialize object on a {backend_name.upper()} device but torch.{backend_name}.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.")
    if hasattr(device_module, 'device_count'):
        device_count = device_module.device_count()
        if device_index >= device_count:
            raise RuntimeError(f'Attempting to deserialize object on {backend_name.upper()} device {device_index} but torch.{backend_name}.device_count() is {device_count}. Please use torch.load with map_location to map your storages to an existing device.')
    return device_index