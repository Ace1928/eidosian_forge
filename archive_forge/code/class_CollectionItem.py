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
class CollectionItem:
    """
    Contains information about an item of a Collection (model, dataset, Space or paper).

    Attributes:
        item_object_id (`str`):
            Unique ID of the item in the collection.
        item_id (`str`):
            ID of the underlying object on the Hub. Can be either a repo_id or a paper id
            e.g. `"jbilcke-hf/ai-comic-factory"`, `"2307.09288"`.
        item_type (`str`):
            Type of the underlying object. Can be one of `"model"`, `"dataset"`, `"space"` or `"paper"`.
        position (`int`):
            Position of the item in the collection.
        note (`str`, *optional*):
            Note associated with the item, as plain text.
    """
    item_object_id: str
    item_id: str
    item_type: str
    position: int
    note: Optional[str] = None

    def __init__(self, _id: str, id: str, type: CollectionItemType_T, position: int, note: Optional[Dict]=None, **kwargs) -> None:
        self.item_object_id: str = _id
        self.item_id: str = id
        self.item_type: CollectionItemType_T = type
        self.position: int = position
        self.note: str = note['text'] if note is not None else None