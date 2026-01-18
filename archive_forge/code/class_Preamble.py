from __future__ import annotations
import re
import typing as t
from dataclasses import dataclass
from enum import auto
from enum import Enum
from ..datastructures import Headers
from ..exceptions import RequestEntityTooLarge
from ..http import parse_options_header
@dataclass(frozen=True)
class Preamble(Event):
    data: bytes