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
def _unpack_dataset_filter(self, dataset_filter: DatasetFilter):
    """
        Unpacks a [`DatasetFilter`] into something readable for `list_datasets`
        """
    dataset_str = ''
    if dataset_filter.author:
        dataset_str = f'{dataset_filter.author}/'
    if dataset_filter.dataset_name:
        dataset_str += dataset_filter.dataset_name
    filter_list = []
    data_attributes = ['benchmark', 'language_creators', 'language', 'multilinguality', 'size_categories', 'task_categories', 'task_ids']
    for attr in data_attributes:
        curr_attr = getattr(dataset_filter, attr)
        if curr_attr is not None:
            if not isinstance(curr_attr, (list, tuple)):
                curr_attr = [curr_attr]
            for data in curr_attr:
                if f'{attr}:' not in data:
                    data = f'{attr}:{data}'
                filter_list.append(data)
    query_dict: Dict[str, Any] = {}
    if dataset_str is not None:
        query_dict['search'] = dataset_str
    query_dict['filter'] = tuple(filter_list)
    return query_dict