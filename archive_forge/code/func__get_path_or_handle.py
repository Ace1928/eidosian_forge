from __future__ import annotations
import io
import json
import os
from typing import (
import warnings
from warnings import catch_warnings
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import _get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
import pandas as pd
from pandas import (
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_string_types_mapper
from pandas.io.common import (
def _get_path_or_handle(path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], fs: Any, storage_options: StorageOptions | None=None, mode: str='rb', is_dir: bool=False) -> tuple[FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any]:
    """File handling for PyArrow."""
    path_or_handle = stringify_path(path)
    if fs is not None:
        pa_fs = import_optional_dependency('pyarrow.fs', errors='ignore')
        fsspec = import_optional_dependency('fsspec', errors='ignore')
        if pa_fs is not None and isinstance(fs, pa_fs.FileSystem):
            if storage_options:
                raise NotImplementedError('storage_options not supported with a pyarrow FileSystem.')
        elif fsspec is not None and isinstance(fs, fsspec.spec.AbstractFileSystem):
            pass
        else:
            raise ValueError(f'filesystem must be a pyarrow or fsspec FileSystem, not a {type(fs).__name__}')
    if is_fsspec_url(path_or_handle) and fs is None:
        if storage_options is None:
            pa = import_optional_dependency('pyarrow')
            pa_fs = import_optional_dependency('pyarrow.fs')
            try:
                fs, path_or_handle = pa_fs.FileSystem.from_uri(path)
            except (TypeError, pa.ArrowInvalid):
                pass
        if fs is None:
            fsspec = import_optional_dependency('fsspec')
            fs, path_or_handle = fsspec.core.url_to_fs(path_or_handle, **storage_options or {})
    elif storage_options and (not is_url(path_or_handle) or mode != 'rb'):
        raise ValueError('storage_options passed with buffer, or non-supported URL')
    handles = None
    if not fs and (not is_dir) and isinstance(path_or_handle, str) and (not os.path.isdir(path_or_handle)):
        handles = get_handle(path_or_handle, mode, is_text=False, storage_options=storage_options)
        fs = None
        path_or_handle = handles.handle
    return (path_or_handle, handles, fs)