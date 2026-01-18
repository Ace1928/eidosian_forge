from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class OpenAIError(Exception):
    pass