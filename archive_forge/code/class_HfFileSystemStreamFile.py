import copy
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
from urllib.parse import quote, unquote
import fsspec
from requests import Response
from ._commit_api import CommitOperationCopy, CommitOperationDelete
from .constants import DEFAULT_REVISION, ENDPOINT, REPO_TYPE_MODEL, REPO_TYPES_MAPPING, REPO_TYPES_URL_PREFIXES
from .file_download import hf_hub_url
from .hf_api import HfApi, LastCommitInfo, RepoFile
from .utils import (
class HfFileSystemStreamFile(fsspec.spec.AbstractBufferedFile):

    def __init__(self, fs: HfFileSystem, path: str, mode: str='rb', revision: Optional[str]=None, block_size: int=0, cache_type: str='none', **kwargs):
        if block_size != 0:
            raise ValueError(f'HfFileSystemStreamFile only supports block_size=0 but got {block_size}')
        if cache_type != 'none':
            raise ValueError(f"HfFileSystemStreamFile only supports cache_type='none' but got {cache_type}")
        if 'w' in mode:
            raise ValueError(f"HfFileSystemStreamFile only supports reading but got mode='{mode}'")
        try:
            self.resolved_path = fs.resolve_path(path, revision=revision)
        except FileNotFoundError as e:
            if 'w' in kwargs.get('mode', ''):
                raise FileNotFoundError(f'{e}.\nMake sure the repository and revision exist before writing data.') from e
        self.details = {'name': self.resolved_path.unresolve(), 'size': None}
        super().__init__(fs, self.resolved_path.unresolve(), mode=mode, block_size=block_size, cache_type=cache_type, **kwargs)
        self.response: Optional[Response] = None
        self.fs: HfFileSystem

    def seek(self, loc: int, whence: int=0):
        if loc == 0 and whence == 1:
            return
        if loc == self.loc and whence == 0:
            return
        raise ValueError('Cannot seek streaming HF file')

    def read(self, length: int=-1):
        read_args = (length,) if length >= 0 else ()
        if self.response is None or self.response.raw.isclosed():
            url = hf_hub_url(repo_id=self.resolved_path.repo_id, revision=self.resolved_path.revision, filename=self.resolved_path.path_in_repo, repo_type=self.resolved_path.repo_type, endpoint=self.fs.endpoint)
            self.response = http_backoff('GET', url, headers=self.fs._api._build_hf_headers(), retry_on_status_codes=(502, 503, 504), stream=True)
            hf_raise_for_status(self.response)
        try:
            out = self.response.raw.read(*read_args)
        except Exception:
            self.response.close()
            url = hf_hub_url(repo_id=self.resolved_path.repo_id, revision=self.resolved_path.revision, filename=self.resolved_path.path_in_repo, repo_type=self.resolved_path.repo_type, endpoint=self.fs.endpoint)
            self.response = http_backoff('GET', url, headers={'Range': 'bytes=%d-' % self.loc, **self.fs._api._build_hf_headers()}, retry_on_status_codes=(502, 503, 504), stream=True)
            hf_raise_for_status(self.response)
            try:
                out = self.response.raw.read(*read_args)
            except Exception:
                self.response.close()
                raise
        self.loc += len(out)
        return out

    def url(self) -> str:
        return self.fs.url(self.path)

    def __del__(self):
        if not hasattr(self, 'resolved_path'):
            return
        return super().__del__()

    def __reduce__(self):
        return (reopen, (self.fs, self.path, self.mode, self.blocksize, self.cache.name))