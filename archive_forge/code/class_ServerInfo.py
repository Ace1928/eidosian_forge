from __future__ import annotations
import logging # isort:skip
from typing import Any, TypedDict
from bokeh import __version__
from ...core.types import ID
from ..message import Message
class ServerInfo(TypedDict):
    version_info: VersionInfo