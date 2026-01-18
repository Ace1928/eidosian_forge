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
class SpaceInfo:
    """
    Contains information about a Space on the Hub.

    <Tip>

    Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made.
    In general, the more specific the query, the more information is returned. On the contrary, when listing spaces
    using [`list_spaces`] only a subset of the attributes are returned.

    </Tip>

    Attributes:
        id (`str`):
            ID of the Space.
        author (`str`, *optional*):
            Author of the Space.
        sha (`str`, *optional*):
            Repo SHA at this particular revision.
        created_at (`datetime`, *optional*):
            Date of creation of the repo on the Hub. Note that the lowest value is `2022-03-02T23:29:04.000Z`,
            corresponding to the date when we began to store creation dates.
        last_modified (`datetime`, *optional*):
            Date of last commit to the repo.
        private (`bool`):
            Is the repo private.
        gated (`Literal["auto", "manual", False]`, *optional*):
            Is the repo gated.
            If so, whether there is manual or automatic approval.
        disabled (`bool`, *optional*):
            Is the Space disabled.
        host (`str`, *optional*):
            Host URL of the Space.
        subdomain (`str`, *optional*):
            Subdomain of the Space.
        likes (`int`):
            Number of likes of the Space.
        tags (`List[str]`):
            List of tags of the Space.
        siblings (`List[RepoSibling]`):
            List of [`huggingface_hub.hf_api.RepoSibling`] objects that constitute the Space.
        card_data (`SpaceCardData`, *optional*):
            Space Card Metadata  as a [`huggingface_hub.repocard_data.SpaceCardData`] object.
        runtime (`SpaceRuntime`, *optional*):
            Space runtime information as a [`huggingface_hub.hf_api.SpaceRuntime`] object.
        sdk (`str`, *optional*):
            SDK used by the Space.
        models (`List[str]`, *optional*):
            List of models used by the Space.
        datasets (`List[str]`, *optional*):
            List of datasets used by the Space.
    """
    id: str
    author: Optional[str]
    sha: Optional[str]
    created_at: Optional[datetime]
    last_modified: Optional[datetime]
    private: bool
    gated: Optional[Literal['auto', 'manual', False]]
    disabled: Optional[bool]
    host: Optional[str]
    subdomain: Optional[str]
    likes: int
    sdk: Optional[str]
    tags: List[str]
    siblings: Optional[List[RepoSibling]]
    card_data: Optional[SpaceCardData]
    runtime: Optional[SpaceRuntime]
    models: Optional[List[str]]
    datasets: Optional[List[str]]

    def __init__(self, **kwargs):
        self.id = kwargs.pop('id')
        self.author = kwargs.pop('author', None)
        self.sha = kwargs.pop('sha', None)
        created_at = kwargs.pop('createdAt', None) or kwargs.pop('created_at', None)
        self.created_at = parse_datetime(created_at) if created_at else None
        last_modified = kwargs.pop('lastModified', None) or kwargs.pop('last_modified', None)
        self.last_modified = parse_datetime(last_modified) if last_modified else None
        self.private = kwargs.pop('private')
        self.gated = kwargs.pop('gated', None)
        self.disabled = kwargs.pop('disabled', None)
        self.host = kwargs.pop('host', None)
        self.subdomain = kwargs.pop('subdomain', None)
        self.likes = kwargs.pop('likes')
        self.sdk = kwargs.pop('sdk', None)
        self.tags = kwargs.pop('tags')
        card_data = kwargs.pop('cardData', None) or kwargs.pop('card_data', None)
        self.card_data = SpaceCardData(**card_data, ignore_metadata_errors=True) if isinstance(card_data, dict) else card_data
        siblings = kwargs.pop('siblings', None)
        self.siblings = [RepoSibling(rfilename=sibling['rfilename'], size=sibling.get('size'), blob_id=sibling.get('blobId'), lfs=BlobLfsInfo(size=sibling['lfs']['size'], sha256=sibling['lfs']['sha256'], pointer_size=sibling['lfs']['pointerSize']) if sibling.get('lfs') else None) for sibling in siblings] if siblings else None
        runtime = kwargs.pop('runtime', None)
        self.runtime = SpaceRuntime(runtime) if runtime else None
        self.models = kwargs.pop('models', None)
        self.datasets = kwargs.pop('datasets', None)
        self.lastModified = self.last_modified
        self.cardData = self.card_data
        self.__dict__.update(**kwargs)