from __future__ import annotations
import logging # isort:skip
from ..application import SessionContext
from .lifecycle import LifecycleHandler

    Calls any on_session_destroyed callbacks defined on the Document
    