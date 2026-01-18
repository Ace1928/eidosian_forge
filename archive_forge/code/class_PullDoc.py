from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, TypedDict
from ...core.types import ID
from ..exceptions import ProtocolError
from ..message import Message
class PullDoc(TypedDict):
    doc: DocJson