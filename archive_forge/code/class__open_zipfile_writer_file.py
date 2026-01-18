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
class _open_zipfile_writer_file(_opener):

    def __init__(self, name) -> None:
        self.file_stream = None
        self.name = str(name)
        try:
            self.name.encode('ascii')
        except UnicodeEncodeError:
            self.file_stream = io.FileIO(self.name, mode='w')
            super().__init__(torch._C.PyTorchFileWriter(self.file_stream))
        else:
            super().__init__(torch._C.PyTorchFileWriter(self.name))

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()