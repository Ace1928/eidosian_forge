from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
class ShowApp(Protocol):

    def __call__(self, app: Application, state: State, notebook_url: str | ProxyUrlFunc, **kw: Any) -> None:
        ...