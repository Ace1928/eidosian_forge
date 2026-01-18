from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import urllib
from pathlib import Path
from typing import Any
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from traitlets.config import LoggingConfigurable
def _list_workspaces(directory: Path, prefix: str) -> list[dict[str, Any]]:
    """
    Return the list of workspaces in a given directory beginning with the
    given prefix.
    """
    workspaces: list = []
    if not directory.exists():
        return workspaces
    items = [item for item in directory.iterdir() if item.name.startswith(prefix) and item.name.endswith(WORKSPACE_EXTENSION)]
    items.sort()
    for slug in items:
        workspace_path: Path = directory / slug
        if workspace_path.exists():
            workspace = _load_with_file_times(workspace_path)
            workspaces.append(workspace)
    return workspaces