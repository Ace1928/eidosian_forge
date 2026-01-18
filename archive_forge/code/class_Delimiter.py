from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from .._compat import DATACLASS_KWARGS
from ..common.utils import isMdAsciiPunct, isPunctChar, isWhiteSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
@dataclass(**DATACLASS_KWARGS)
class Delimiter:
    marker: int
    length: int
    token: int
    end: int
    open: bool
    close: bool
    level: bool | None = None