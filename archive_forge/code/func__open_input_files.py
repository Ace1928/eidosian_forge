from __future__ import annotations
import json
from typing import Protocol, runtime_checkable
from uuid import uuid4
import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from packaging.version import parse as parse_version
def _open_input_files(paths, fs=None, context_stack=None, open_file_func=None, precache_options=None, **kwargs):
    """Return a list of open-file objects given
    a list of input-file paths.

    WARNING: This utility is experimental, and is meant
    for internal ``dask.dataframe`` use only.

    Parameters
    ----------
    paths : list(str)
        Remote or local path of the parquet file
    fs : fsspec object, optional
        File-system instance to use for file handling
    context_stack : contextlib.ExitStack, Optional
        Context manager to use for open files.
    open_file_func : callable, optional
        Callable function to use for file opening. If this argument
        is specified, ``open_file_func(path, **kwargs)`` will be used
        to open each file in ``paths``. Default is ``fs.open``.
    precache_options : dict, optional
        Dictionary of key-word arguments to use for precaching.
        If ``precache_options`` contains ``{"method": "parquet"}``,
        ``fsspec.parquet.open_parquet_file`` will be used for remote
        storage.
    **kwargs :
        Key-word arguments to pass to the appropriate open function
    """
    if open_file_func is not None:
        return [_set_context(open_file_func(path, **kwargs), context_stack) for path in paths]
    precache_options = (precache_options or {}).copy()
    precache = precache_options.pop('method', None)
    if precache == 'parquet' and fs is not None and (not _is_local_fs(fs)) and (parse_version(fsspec.__version__) > parse_version('2021.11.0')):
        kwargs.update(precache_options)
        row_groups = kwargs.pop('row_groups', None) or [None] * len(paths)
        cache_type = kwargs.pop('cache_type', 'parts')
        if cache_type != 'parts':
            raise ValueError(f"'parts' `cache_type` required for 'parquet' precaching, got {cache_type}.")
        return [_set_context(fsspec_parquet.open_parquet_file(path, fs=fs, row_groups=rgs, **kwargs), context_stack) for path, rgs in zip(paths, row_groups)]
    elif fs is not None:
        return [_set_context(fs.open(path, **kwargs), context_stack) for path in paths]
    return [_set_context(open(path, **kwargs), context_stack) for path in paths]