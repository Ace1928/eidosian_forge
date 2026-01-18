from __future__ import annotations
import logging  # isort:skip
import json
import os
import re
from os.path import relpath
from pathlib import Path
from typing import (
from . import __version__
from .core.templates import CSS_RESOURCES, JS_RESOURCES
from .core.types import ID, PathLike
from .model import Model
from .settings import LogLevel, settings
from .util.dataclasses import dataclass, field
from .util.paths import ROOT_DIR
from .util.token import generate_session_id
from .util.version import is_full_release
def _get_server_urls(root_url: str=DEFAULT_SERVER_HTTP_URL, minified: bool=True, path_versioner: PathVersioner | None=None) -> Urls:
    _minified = '.min' if minified else ''

    def mk_url(comp: str, kind: Kind) -> str:
        path = f'{kind}/{comp}{_minified}.{kind}'
        if path_versioner is not None:
            path = path_versioner(path)
        return f'{root_url}static/{path}'
    return Urls(urls=lambda components, kind: [mk_url(component, kind) for component in components])