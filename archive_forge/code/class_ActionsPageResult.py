from __future__ import annotations
import time
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Action, ActionFailedException, ActionTimeoutException
class ActionsPageResult(NamedTuple):
    actions: list[BoundAction]
    meta: Meta | None