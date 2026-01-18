from __future__ import annotations
import io
import os
import pathlib
from typing import overload
from typing_extensions import TypeGuard
import anyio
from ._types import (
from ._utils import is_tuple_t, is_mapping_t, is_sequence_t
def _read_file_content(file: FileContent) -> HttpxFileContent:
    if isinstance(file, os.PathLike):
        return pathlib.Path(file).read_bytes()
    return file