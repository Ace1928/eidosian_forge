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
def _build_hf_headers(self, token: Optional[Union[bool, str]]=None, is_write_action: bool=False, library_name: Optional[str]=None, library_version: Optional[str]=None, user_agent: Union[Dict, str, None]=None) -> Dict[str, str]:
    """
        Alias for [`build_hf_headers`] that uses the token from [`HfApi`] client
        when `token` is not provided.
        """
    if token is None:
        token = self.token
    return build_hf_headers(token=token, is_write_action=is_write_action, library_name=library_name or self.library_name, library_version=library_version or self.library_version, user_agent=user_agent or self.user_agent)