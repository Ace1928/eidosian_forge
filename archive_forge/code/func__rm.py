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
def _rm(self, path: str, revision: Optional[str]=None, **kwargs) -> None:
    resolved_path = self.resolve_path(path, revision=revision)
    self._api.delete_file(path_in_repo=resolved_path.path_in_repo, repo_id=resolved_path.repo_id, token=self.token, repo_type=resolved_path.repo_type, revision=resolved_path.revision, commit_message=kwargs.get('commit_message'), commit_description=kwargs.get('commit_description'))
    self.invalidate_cache(path=resolved_path.unresolve())