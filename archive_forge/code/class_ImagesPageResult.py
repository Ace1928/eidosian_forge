from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Image
class ImagesPageResult(NamedTuple):
    images: list[BoundImage]
    meta: Meta | None