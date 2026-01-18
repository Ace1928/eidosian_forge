from __future__ import annotations
import asyncio
import datetime as dt
import importlib
import inspect
import logging
import os
import pathlib
import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from functools import partial, wraps
from html import escape
from types import FunctionType, MethodType
from typing import (
from urllib.parse import urljoin, urlparse
import bokeh
import param
import tornado
from bokeh.application import Application as BkApplication
from bokeh.application.handlers.function import FunctionHandler
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import AUTOLOAD_JS, FILE, MACROS
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
from bokeh.embed.bundle import Script
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import RenderItem
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.models import CustomJS
from bokeh.server.server import Server as BokehServer
from bokeh.server.urls import per_app_patterns, toplevel_patterns
from bokeh.server.views.autoload_js_handler import (
from bokeh.server.views.doc_handler import DocHandler as BkDocHandler
from bokeh.server.views.root_handler import RootHandler as BkRootHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.util.serialization import make_id
from bokeh.util.token import (
from tornado.ioloop import IOLoop
from tornado.web import (
from tornado.wsgi import WSGIContainer
from ..config import config
from ..util import edit_readonly, fullpath
from ..util.warnings import warn
from .application import Application, build_single_handler_application
from .document import (  # noqa
from .liveness import LivenessHandler
from .loading import LOADING_INDICATOR_CSS_CLASS
from .logging import (
from .reload import record_modules
from .resources import (
from .session import generate_session
from .state import set_curdoc, state
def _generate_token_payload(self):
    app = self.application
    if app.include_headers is None:
        excluded_headers = app.exclude_headers or []
        allowed_headers = [header for header in self.request.headers if header not in excluded_headers]
    else:
        allowed_headers = app.include_headers
    headers = {k: v for k, v in self.request.headers.items() if k in allowed_headers}
    if app.include_cookies is None:
        excluded_cookies = app.exclude_cookies or []
        allowed_cookies = [cookie for cookie in self.request.cookies if cookie not in excluded_cookies]
    else:
        allowed_cookies = app.include_cookies
    cookies = {k: v.value for k, v in self.request.cookies.items() if k in allowed_cookies}
    if cookies and 'Cookie' in headers and ('Cookie' not in (app.include_headers or [])):
        del headers['Cookie']
    arguments = {} if self.request.arguments is None else self.request.arguments
    payload = {'headers': headers, 'cookies': cookies, 'arguments': arguments}
    payload.update(self.application_context.application.process_request(self.request))
    return payload