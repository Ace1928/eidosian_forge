from __future__ import annotations
import io
import os
import pathlib
from typing import overload
from typing_extensions import TypeGuard
import anyio
from ._types import (
from ._utils import is_tuple_t, is_mapping_t, is_sequence_t
def _transform_file(file: FileTypes) -> HttpxFileTypes:
    if is_file_content(file):
        if isinstance(file, os.PathLike):
            path = pathlib.Path(file)
            return (path.name, path.read_bytes())
        return file
    if is_tuple_t(file):
        return (file[0], _read_file_content(file[1]), *file[2:])
    raise TypeError(f'Expected file types input to be a FileContent type or to be a tuple')