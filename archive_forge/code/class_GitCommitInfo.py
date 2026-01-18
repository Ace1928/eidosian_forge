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
class GitCommitInfo:
    """
    Contains information about a git commit for a repo on the Hub. Check out [`list_repo_commits`] for more details.

    Attributes:
        commit_id (`str`):
            OID of the commit (e.g. `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`)
        authors (`List[str]`):
            List of authors of the commit.
        created_at (`datetime`):
            Datetime when the commit was created.
        title (`str`):
            Title of the commit. This is a free-text value entered by the authors.
        message (`str`):
            Description of the commit. This is a free-text value entered by the authors.
        formatted_title (`str`):
            Title of the commit formatted as HTML. Only returned if `formatted=True` is set.
        formatted_message (`str`):
            Description of the commit formatted as HTML. Only returned if `formatted=True` is set.
    """
    commit_id: str
    authors: List[str]
    created_at: datetime
    title: str
    message: str
    formatted_title: Optional[str]
    formatted_message: Optional[str]