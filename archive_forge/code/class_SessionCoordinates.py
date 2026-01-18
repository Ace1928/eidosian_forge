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
class SessionCoordinates:
    """ Internal class used to parse kwargs for server URL, app_path, and session_id."""
    _url: str
    _session_id: ID | None

    def __init__(self, *, url: str=DEFAULT_SERVER_HTTP_URL, session_id: ID | None=None) -> None:
        self._url = url
        if self._url == 'default':
            self._url = DEFAULT_SERVER_HTTP_URL
        if self._url.startswith('ws'):
            raise ValueError('url should be the http or https URL for the server, not the websocket URL')
        self._url = self._url.rstrip('/')
        self._session_id = session_id

    @property
    def url(self) -> str:
        return self._url

    @property
    def session_id(self) -> ID:
        """ Session ID derived from the kwargs provided."""
        if self._session_id is None:
            self._session_id = generate_session_id()
        return self._session_id

    @property
    def session_id_allowing_none(self) -> ID | None:
        """ Session ID provided in kwargs, keeping it None if it hasn't been generated yet.

        The purpose of this is to preserve ``None`` as long as possible... in some cases
        we may never generate the session ID because we generate it on the server.
        """
        return self._session_id