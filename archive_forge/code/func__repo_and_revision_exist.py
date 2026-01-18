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
def _repo_and_revision_exist(self, repo_type: str, repo_id: str, revision: Optional[str]) -> Tuple[bool, Optional[Exception]]:
    if (repo_type, repo_id, revision) not in self._repo_and_revision_exists_cache:
        try:
            self._api.repo_info(repo_id, revision=revision, repo_type=repo_type)
        except (RepositoryNotFoundError, HFValidationError) as e:
            self._repo_and_revision_exists_cache[repo_type, repo_id, revision] = (False, e)
            self._repo_and_revision_exists_cache[repo_type, repo_id, None] = (False, e)
        except RevisionNotFoundError as e:
            self._repo_and_revision_exists_cache[repo_type, repo_id, revision] = (False, e)
            self._repo_and_revision_exists_cache[repo_type, repo_id, None] = (True, None)
        else:
            self._repo_and_revision_exists_cache[repo_type, repo_id, revision] = (True, None)
            self._repo_and_revision_exists_cache[repo_type, repo_id, None] = (True, None)
    return self._repo_and_revision_exists_cache[repo_type, repo_id, revision]