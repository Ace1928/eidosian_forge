import glob
import io
import os
import posixpath
import re
import tarfile
import time
import xml.dom.minidom
import zipfile
from asyncio import TimeoutError
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import fsspec
from aiohttp.client_exceptions import ClientError
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from .. import config
from ..filesystems import COMPRESSION_FILESYSTEMS
from ..utils.file_utils import (
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
class xPath(type(Path())):
    """Extension of `pathlib.Path` to support both local paths and remote URLs."""

    def __str__(self):
        path_str = super().__str__()
        main_hop, *rest_hops = path_str.split('::')
        if is_local_path(main_hop):
            return main_hop
        path_as_posix = path_str.replace('\\', '/')
        path_as_posix = SINGLE_SLASH_AFTER_PROTOCOL_PATTERN.sub('://', path_as_posix)
        path_as_posix += '//' if path_as_posix.endswith(':') else ''
        return path_as_posix

    def exists(self, download_config: Optional[DownloadConfig]=None):
        """Extend `pathlib.Path.exists` method to support both local and remote files.

        Args:
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Returns:
            `bool`
        """
        return xexists(str(self), download_config=download_config)

    def glob(self, pattern, download_config: Optional[DownloadConfig]=None):
        """Glob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Yields:
            [`xPath`]
        """
        posix_path = self.as_posix()
        main_hop, *rest_hops = posix_path.split('::')
        if is_local_path(main_hop):
            yield from Path(main_hop).glob(pattern)
        else:
            if rest_hops:
                urlpath = rest_hops[0]
                urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
                storage_options = {urlpath.split('://')[0]: storage_options}
                posix_path = '::'.join([main_hop, urlpath, *rest_hops[1:]])
            else:
                storage_options = None
            fs, *_ = fsspec.get_fs_token_paths(xjoin(posix_path, pattern), storage_options=storage_options)
            globbed_paths = fs.glob(xjoin(main_hop, pattern))
            for globbed_path in globbed_paths:
                yield type(self)('::'.join([f'{fs.protocol}://{globbed_path}'] + rest_hops))

    def rglob(self, pattern, **kwargs):
        """Rglob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.

        Yields:
            [`xPath`]
        """
        return self.glob('**/' + pattern, **kwargs)

    @property
    def parent(self) -> 'xPath':
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            [`xPath`]
        """
        return type(self)(xdirname(self.as_posix()))

    @property
    def name(self) -> str:
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split('::')[0]).name

    @property
    def stem(self) -> str:
        """Stem function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split('::')[0]).stem

    @property
    def suffix(self) -> str:
        """Suffix function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split('::')[0]).suffix

    def open(self, *args, **kwargs):
        """Extend :func:`xopen` to support argument of type :obj:`~pathlib.Path`.

        Args:
            **args: Arguments passed to :func:`fsspec.open`.
            **kwargs: Keyword arguments passed to :func:`fsspec.open`.

        Returns:
            `io.FileIO`: File-like object.
        """
        return xopen(str(self), *args, **kwargs)

    def joinpath(self, *p: Tuple[str, ...]) -> 'xPath':
        """Extend :func:`xjoin` to support argument of type :obj:`~pathlib.Path`.

        Args:
            *p (`tuple` of `str`): Other path components.

        Returns:
            [`xPath`]
        """
        return type(self)(xjoin(self.as_posix(), *p))

    def __truediv__(self, p: str) -> 'xPath':
        return self.joinpath(p)

    def with_suffix(self, suffix):
        main_hop, *rest_hops = str(self).split('::')
        if is_local_path(main_hop):
            return type(self)(str(super().with_suffix(suffix)))
        return type(self)('::'.join([type(self)(PurePosixPath(main_hop).with_suffix(suffix)).as_posix()] + rest_hops))