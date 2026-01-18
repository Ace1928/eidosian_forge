from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _find_absolute_paths(paths: str | os.PathLike | NestedSequence[str | os.PathLike], **kwargs) -> list[str]:
    """
    Find absolute paths from the pattern.

    Parameters
    ----------
    paths :
        Path(s) to file(s). Can include wildcards like * .
    **kwargs :
        Extra kwargs. Mainly for fsspec.

    Examples
    --------
    >>> from pathlib import Path

    >>> directory = Path(xr.backends.common.__file__).parent
    >>> paths = str(Path(directory).joinpath("comm*n.py"))  # Find common with wildcard
    >>> paths = xr.backends.common._find_absolute_paths(paths)
    >>> [Path(p).name for p in paths]
    ['common.py']
    """
    if isinstance(paths, str):
        if is_remote_uri(paths) and kwargs.get('engine', None) == 'zarr':
            try:
                from fsspec.core import get_fs_token_paths
            except ImportError as e:
                raise ImportError('The use of remote URLs for opening zarr requires the package fsspec') from e
            fs, _, _ = get_fs_token_paths(paths, mode='rb', storage_options=kwargs.get('backend_kwargs', {}).get('storage_options', {}), expand=False)
            tmp_paths = fs.glob(fs._strip_protocol(paths))
            paths = [fs.get_mapper(path) for path in tmp_paths]
        elif is_remote_uri(paths):
            raise ValueError(f"cannot do wild-card matching for paths that are remote URLs unless engine='zarr' is specified. Got paths: {paths}. Instead, supply paths as an explicit list of strings.")
        else:
            paths = sorted(glob(_normalize_path(paths)))
    elif isinstance(paths, os.PathLike):
        paths = [os.fspath(paths)]
    else:
        paths = [os.fspath(p) if isinstance(p, os.PathLike) else p for p in paths]
    return paths