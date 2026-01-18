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
@observe(*list(_deprecated_traits))
def _deprecated_trait(self, change):
    """observer for deprecated traits"""
    old_attr = change.name
    new_attr, version = self._deprecated_traits[old_attr]
    new_value = getattr(self, new_attr)
    if new_value != change.new:
        self.log.warning(f'{self.__class__.__name__}.{old_attr} is deprecated in jupyter_server {version}, use {self.__class__.__name__}.{new_attr} instead')
        setattr(self, new_attr, change.new)