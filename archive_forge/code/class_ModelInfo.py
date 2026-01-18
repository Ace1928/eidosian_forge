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
class ModelInfo:
    """
    Contains information about a model on the Hub.

    <Tip>

    Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made.
    In general, the more specific the query, the more information is returned. On the contrary, when listing models
    using [`list_models`] only a subset of the attributes are returned.

    </Tip>

    Attributes:
        id (`str`):
            ID of model.
        author (`str`, *optional*):
            Author of the model.
        sha (`str`, *optional*):
            Repo SHA at this particular revision.
        created_at (`datetime`, *optional*):
            Date of creation of the repo on the Hub. Note that the lowest value is `2022-03-02T23:29:04.000Z`,
            corresponding to the date when we began to store creation dates.
        last_modified (`datetime`, *optional*):
            Date of last commit to the repo.
        private (`bool`):
            Is the repo private.
        disabled (`bool`, *optional*):
            Is the repo disabled.
        gated (`Literal["auto", "manual", False]`, *optional*):
            Is the repo gated.
            If so, whether there is manual or automatic approval.
        downloads (`int`):
            Number of downloads of the model.
        likes (`int`):
            Number of likes of the model.
        library_name (`str`, *optional*):
            Library associated with the model.
        tags (`List[str]`):
            List of tags of the model. Compared to `card_data.tags`, contains extra tags computed by the Hub
            (e.g. supported libraries, model's arXiv).
        pipeline_tag (`str`, *optional*):
            Pipeline tag associated with the model.
        mask_token (`str`, *optional*):
            Mask token used by the model.
        widget_data (`Any`, *optional*):
            Widget data associated with the model.
        model_index (`Dict`, *optional*):
            Model index for evaluation.
        config (`Dict`, *optional*):
            Model configuration.
        transformers_info (`TransformersInfo`, *optional*):
            Transformers-specific info (auto class, processor, etc.) associated with the model.
        card_data (`ModelCardData`, *optional*):
            Model Card Metadata  as a [`huggingface_hub.repocard_data.ModelCardData`] object.
        siblings (`List[RepoSibling]`):
            List of [`huggingface_hub.hf_api.RepoSibling`] objects that constitute the model.
        spaces (`List[str]`, *optional*):
            List of spaces using the model.
        safetensors (`SafeTensorsInfo`, *optional*):
            Model's safetensors information.
    """
    id: str
    author: Optional[str]
    sha: Optional[str]
    created_at: Optional[datetime]
    last_modified: Optional[datetime]
    private: bool
    gated: Optional[Literal['auto', 'manual', False]]
    disabled: Optional[bool]
    downloads: int
    likes: int
    library_name: Optional[str]
    tags: List[str]
    pipeline_tag: Optional[str]
    mask_token: Optional[str]
    card_data: Optional[ModelCardData]
    widget_data: Optional[Any]
    model_index: Optional[Dict]
    config: Optional[Dict]
    transformers_info: Optional[TransformersInfo]
    siblings: Optional[List[RepoSibling]]
    spaces: Optional[List[str]]
    safetensors: Optional[SafeTensorsInfo]

    def __init__(self, **kwargs):
        self.id = kwargs.pop('id')
        self.author = kwargs.pop('author', None)
        self.sha = kwargs.pop('sha', None)
        last_modified = kwargs.pop('lastModified', None) or kwargs.pop('last_modified', None)
        self.last_modified = parse_datetime(last_modified) if last_modified else None
        created_at = kwargs.pop('createdAt', None) or kwargs.pop('created_at', None)
        self.created_at = parse_datetime(created_at) if created_at else None
        self.private = kwargs.pop('private')
        self.gated = kwargs.pop('gated', None)
        self.disabled = kwargs.pop('disabled', None)
        self.downloads = kwargs.pop('downloads')
        self.likes = kwargs.pop('likes')
        self.library_name = kwargs.pop('library_name', None)
        self.tags = kwargs.pop('tags')
        self.pipeline_tag = kwargs.pop('pipeline_tag', None)
        self.mask_token = kwargs.pop('mask_token', None)
        card_data = kwargs.pop('cardData', None) or kwargs.pop('card_data', None)
        self.card_data = ModelCardData(**card_data, ignore_metadata_errors=True) if isinstance(card_data, dict) else card_data
        self.widget_data = kwargs.pop('widgetData', None)
        self.model_index = kwargs.pop('model-index', None) or kwargs.pop('model_index', None)
        self.config = kwargs.pop('config', None)
        transformers_info = kwargs.pop('transformersInfo', None) or kwargs.pop('transformers_info', None)
        self.transformers_info = TransformersInfo(**transformers_info) if transformers_info else None
        siblings = kwargs.pop('siblings', None)
        self.siblings = [RepoSibling(rfilename=sibling['rfilename'], size=sibling.get('size'), blob_id=sibling.get('blobId'), lfs=BlobLfsInfo(size=sibling['lfs']['size'], sha256=sibling['lfs']['sha256'], pointer_size=sibling['lfs']['pointerSize']) if sibling.get('lfs') else None) for sibling in siblings] if siblings else None
        self.spaces = kwargs.pop('spaces', None)
        safetensors = kwargs.pop('safetensors', None)
        self.safetensors = SafeTensorsInfo(**safetensors) if safetensors else None
        self.lastModified = self.last_modified
        self.cardData = self.card_data
        self.transformersInfo = self.transformers_info
        self.__dict__.update(**kwargs)