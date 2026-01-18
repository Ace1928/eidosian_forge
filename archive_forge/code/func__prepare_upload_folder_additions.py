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
def _prepare_upload_folder_additions(folder_path: Union[str, Path], path_in_repo: str, allow_patterns: Optional[Union[List[str], str]]=None, ignore_patterns: Optional[Union[List[str], str]]=None) -> List[CommitOperationAdd]:
    """Generate the list of Add operations for a commit to upload a folder.

    Files not matching the `allow_patterns` (allowlist) and `ignore_patterns` (denylist)
    constraints are discarded.
    """
    folder_path = Path(folder_path).expanduser().resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")
    relpath_to_abspath = {path.relative_to(folder_path).as_posix(): path for path in sorted(folder_path.glob('**/*')) if path.is_file()}
    prefix = f'{path_in_repo.strip('/')}/' if path_in_repo else ''
    return [CommitOperationAdd(path_or_fileobj=relpath_to_abspath[relpath], path_in_repo=prefix + relpath) for relpath in filter_repo_objects(relpath_to_abspath.keys(), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)]