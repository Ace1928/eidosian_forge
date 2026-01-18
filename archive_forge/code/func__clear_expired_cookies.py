from __future__ import annotations
import asyncio
import json
import logging
import os
import typing as ty
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from socket import gaierror
from jupyter_events import EventLogger
from tornado import web
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPResponse
from traitlets import (
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
def _clear_expired_cookies(self) -> None:
    """Clear expired cookies."""
    check_time = datetime.now(tz=timezone.utc)
    expired_keys = []
    for key, (morsel, store_time) in self._cookies.items():
        cookie_max_age = morsel.get('max-age')
        if not cookie_max_age:
            continue
        expired_timedelta = check_time - store_time
        if expired_timedelta.total_seconds() > float(cookie_max_age):
            expired_keys.append(key)
    for key in expired_keys:
        self._cookies.pop(key)