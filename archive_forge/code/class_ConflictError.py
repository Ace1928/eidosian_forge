from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class ConflictError(APIStatusError):
    status_code: Literal[409] = 409