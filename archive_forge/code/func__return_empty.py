from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from .handler import Handler
def _return_empty(request: HTTPServerRequest) -> dict[str, Any]:
    return {}