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
def _parse_revision_from_pr_url(pr_url: str) -> str:
    """Safely parse revision number from a PR url.

    Example:
    ```py
    >>> _parse_revision_from_pr_url("https://huggingface.co/bigscience/bloom/discussions/2")
    "refs/pr/2"
    ```
    """
    re_match = re.match(_REGEX_DISCUSSION_URL, pr_url)
    if re_match is None:
        raise RuntimeError(f"Unexpected response from the hub, expected a Pull Request URL but got: '{pr_url}'")
    return f'refs/pr/{re_match[1]}'