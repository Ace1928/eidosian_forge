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
def _unpack_model_filter(self, model_filter: ModelFilter):
    """
        Unpacks a [`ModelFilter`] into something readable for `list_models`
        """
    model_str = ''
    if model_filter.author:
        model_str = f'{model_filter.author}/'
    if model_filter.model_name:
        model_str += model_filter.model_name
    filter_list: List[str] = []
    if model_filter.task:
        filter_list.extend([model_filter.task] if isinstance(model_filter.task, str) else model_filter.task)
    if model_filter.trained_dataset:
        if not isinstance(model_filter.trained_dataset, (list, tuple)):
            model_filter.trained_dataset = [model_filter.trained_dataset]
        for dataset in model_filter.trained_dataset:
            if 'dataset:' not in dataset:
                dataset = f'dataset:{dataset}'
            filter_list.append(dataset)
    if model_filter.library:
        filter_list.extend([model_filter.library] if isinstance(model_filter.library, str) else model_filter.library)
    if model_filter.tags:
        filter_list.extend([model_filter.tags] if isinstance(model_filter.tags, str) else model_filter.tags)
    query_dict: Dict[str, Any] = {}
    if model_str:
        query_dict['search'] = model_str
    if isinstance(model_filter.language, list):
        filter_list.extend(model_filter.language)
    elif isinstance(model_filter.language, str):
        filter_list.append(model_filter.language)
    query_dict['filter'] = tuple(filter_list)
    return query_dict