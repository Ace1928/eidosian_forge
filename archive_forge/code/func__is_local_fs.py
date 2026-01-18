from __future__ import annotations
import json
from typing import Protocol, runtime_checkable
from uuid import uuid4
import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from packaging.version import parse as parse_version
def _is_local_fs(fs):
    """Check if an fsspec file-system is local"""
    return fs and (isinstance(fs, LocalFileSystem) or _is_local_fs_pyarrow(fs))