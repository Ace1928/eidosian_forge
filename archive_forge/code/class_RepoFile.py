from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
@dataclass
class RepoFile:
    """
    Contains information about a file on the Hub.

    Attributes:
        path (str):
            file path relative to the repo root.
        size (`int`):
            The file's size, in bytes.
        blob_id (`str`):
            The file's git OID.
        lfs (`BlobLfsInfo`):
            The file's LFS metadata.
        last_commit (`LastCommitInfo`, *optional*):
            The file's last commit metadata. Only defined if [`list_files_info`], [`list_repo_tree`] and [`get_paths_info`]
            are called with `expand=True`.
        security (`BlobSecurityInfo`, *optional*):
            The file's security scan metadata. Only defined if [`list_files_info`], [`list_repo_tree`] and [`get_paths_info`]
            are called with `expand=True`.
    """
    path: str
    size: int
    blob_id: str
    lfs: Optional[BlobLfsInfo] = None
    last_commit: Optional[LastCommitInfo] = None
    security: Optional[BlobSecurityInfo] = None

    def __init__(self, **kwargs):
        self.path = kwargs.pop('path')
        self.size = kwargs.pop('size')
        self.blob_id = kwargs.pop('oid')
        lfs = kwargs.pop('lfs', None)
        if lfs is not None:
            lfs = BlobLfsInfo(size=lfs['size'], sha256=lfs['oid'], pointer_size=lfs['pointerSize'])
        self.lfs = lfs
        last_commit = kwargs.pop('lastCommit', None) or kwargs.pop('last_commit', None)
        if last_commit is not None:
            last_commit = LastCommitInfo(oid=last_commit['id'], title=last_commit['title'], date=parse_datetime(last_commit['date']))
        self.last_commit = last_commit
        security = kwargs.pop('security', None)
        if security is not None:
            security = BlobSecurityInfo(safe=security['safe'], av_scan=security['avScan'], pickle_import_scan=security['pickleImportScan'])
        self.security = security
        self.rfilename = self.path
        self.lastCommit = self.last_commit