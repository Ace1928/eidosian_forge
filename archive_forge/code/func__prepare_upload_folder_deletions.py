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
def _prepare_upload_folder_deletions(self, repo_id: str, repo_type: Optional[str], revision: Optional[str], token: Optional[str], path_in_repo: str, delete_patterns: Optional[Union[List[str], str]]) -> List[CommitOperationDelete]:
    """Generate the list of Delete operations for a commit to delete files from a repo.

        List remote files and match them against the `delete_patterns` constraints. Returns a list of [`CommitOperationDelete`]
        with the matching items.

        Note: `.gitattributes` file is essential to make a repo work properly on the Hub. This file will always be
              kept even if it matches the `delete_patterns` constraints.
        """
    if delete_patterns is None:
        return []
    filenames = self.list_repo_files(repo_id=repo_id, revision=revision, repo_type=repo_type, token=token)
    if path_in_repo:
        path_in_repo = path_in_repo.strip('/') + '/'
        relpath_to_abspath = {file[len(path_in_repo):]: file for file in filenames if file.startswith(path_in_repo)}
    else:
        relpath_to_abspath = {file: file for file in filenames}
    return [CommitOperationDelete(path_in_repo=relpath_to_abspath[relpath], is_folder=False) for relpath in filter_repo_objects(relpath_to_abspath.keys(), allow_patterns=delete_patterns) if relpath_to_abspath[relpath] != '.gitattributes']