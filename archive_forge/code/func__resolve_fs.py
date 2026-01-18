from __future__ import annotations
import inspect
import logging
import os
import shutil
import uuid
from typing import Optional
from .asyn import AsyncFileSystem, _run_coros_in_chunks, sync_wrapper
from .callbacks import DEFAULT_CALLBACK
from .core import filesystem, get_filesystem_class, split_protocol, url_to_fs
def _resolve_fs(url, method=None, protocol=None, storage_options=None):
    """Pick instance of backend FS"""
    method = method or default_method
    protocol = protocol or split_protocol(url)[0]
    storage_options = storage_options or {}
    if method == 'default':
        return filesystem(protocol)
    if method == 'generic':
        return _generic_fs[protocol]
    if method == 'current':
        cls = get_filesystem_class(protocol)
        return cls.current()
    if method == 'options':
        fs, _ = url_to_fs(url, **storage_options.get(protocol, {}))
        return fs
    raise ValueError(f'Unknown FS resolution method: {method}')