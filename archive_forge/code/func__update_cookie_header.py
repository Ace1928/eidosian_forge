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
def _update_cookie_header(self, connection_args: dict[str, ty.Any]) -> None:
    """Update a cookie header."""
    self._clear_expired_cookies()
    gateway_cookie_values = '; '.join((f'{name}={morsel.coded_value}' for name, (morsel, _time) in self._cookies.items()))
    if gateway_cookie_values:
        headers = connection_args.get('headers', {})
        cookie_header_name = next((header_key for header_key in headers if header_key.lower() == 'cookie'), 'Cookie')
        existing_cookie = headers.get(cookie_header_name)
        if existing_cookie:
            gateway_cookie_values = existing_cookie + '; ' + gateway_cookie_values
        headers[cookie_header_name] = gateway_cookie_values
        connection_args['headers'] = headers