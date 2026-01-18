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
def _upload_chunk(self, final: bool=False) -> None:
    self.buffer.seek(0)
    block = self.buffer.read()
    self.temp_file.write(block)
    if final:
        self.temp_file.close()
        self.fs._api.upload_file(path_or_fileobj=self.temp_file.name, path_in_repo=self.resolved_path.path_in_repo, repo_id=self.resolved_path.repo_id, token=self.fs.token, repo_type=self.resolved_path.repo_type, revision=self.resolved_path.revision, commit_message=self.kwargs.get('commit_message'), commit_description=self.kwargs.get('commit_description'))
        os.remove(self.temp_file.name)
        self.fs.invalidate_cache(path=self.resolved_path.unresolve())