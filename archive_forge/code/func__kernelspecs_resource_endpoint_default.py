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
@default('kernelspecs_resource_endpoint')
def _kernelspecs_resource_endpoint_default(self):
    return os.environ.get(self.kernelspecs_resource_endpoint_env, self.kernelspecs_resource_endpoint_default_value)