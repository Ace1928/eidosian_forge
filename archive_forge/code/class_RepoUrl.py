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
class RepoUrl(str):
    """Subclass of `str` describing a repo URL on the Hub.

    `RepoUrl` is returned by `HfApi.create_repo`. It inherits from `str` for backward
    compatibility. At initialization, the URL is parsed to populate properties:
    - endpoint (`str`)
    - namespace (`Optional[str]`)
    - repo_name (`str`)
    - repo_id (`str`)
    - repo_type (`Literal["model", "dataset", "space"]`)
    - url (`str`)

    Args:
        url (`Any`):
            String value of the repo url.
        endpoint (`str`, *optional*):
            Endpoint of the Hub. Defaults to <https://huggingface.co>.

    Example:
    ```py
    >>> RepoUrl('https://huggingface.co/gpt2')
    RepoUrl('https://huggingface.co/gpt2', endpoint='https://huggingface.co', repo_type='model', repo_id='gpt2')

    >>> RepoUrl('https://hub-ci.huggingface.co/datasets/dummy_user/dummy_dataset', endpoint='https://hub-ci.huggingface.co')
    RepoUrl('https://hub-ci.huggingface.co/datasets/dummy_user/dummy_dataset', endpoint='https://hub-ci.huggingface.co', repo_type='dataset', repo_id='dummy_user/dummy_dataset')

    >>> RepoUrl('hf://datasets/my-user/my-dataset')
    RepoUrl('hf://datasets/my-user/my-dataset', endpoint='https://huggingface.co', repo_type='dataset', repo_id='user/dataset')

    >>> HfApi.create_repo("dummy_model")
    RepoUrl('https://huggingface.co/Wauplin/dummy_model', endpoint='https://huggingface.co', repo_type='model', repo_id='Wauplin/dummy_model')
    ```

    Raises:
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If URL cannot be parsed.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `repo_type` is unknown.
    """

    def __new__(cls, url: Any, endpoint: Optional[str]=None):
        return super(RepoUrl, cls).__new__(cls, url)

    def __init__(self, url: Any, endpoint: Optional[str]=None) -> None:
        super().__init__()
        self.endpoint = endpoint or ENDPOINT
        repo_type, namespace, repo_name = repo_type_and_id_from_hf_id(self, hub_url=self.endpoint)
        self.namespace = namespace
        self.repo_name = repo_name
        self.repo_id = repo_name if namespace is None else f'{namespace}/{repo_name}'
        self.repo_type = repo_type or REPO_TYPE_MODEL
        self.url = str(self)

    def __repr__(self) -> str:
        return f"RepoUrl('{self}', endpoint='{self.endpoint}', repo_type='{self.repo_type}', repo_id='{self.repo_id}')"