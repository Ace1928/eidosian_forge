from __future__ import annotations
import re
from typing import Protocol
from ..common.utils import arrayReplaceAt, isLinkClose, isLinkOpen
from ..token import Token
from .state_core import StateCore
class _LinkType(Protocol):
    url: str
    text: str
    index: int
    last_index: int
    schema: str | None