from __future__ import annotations
import logging # isort:skip
from typing import Any, Callable
from ...document import Document
from ..application import ServerContext, SessionContext
from .handler import Handler
def _do_nothing(ignored: Any) -> None:
    pass