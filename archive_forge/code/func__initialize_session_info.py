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
def _initialize_session_info(session_context: 'BokehSessionContext'):
    from ..config import config
    session_id = session_context.id
    sessions = state.session_info['sessions']
    history = -1 if config._admin else config.session_history
    if not config._admin and (history == 0 or session_id in sessions):
        return
    state.session_info['total'] += 1
    if history > 0 and len(sessions) >= history:
        old_history = list(sessions.items())
        sessions = dict(old_history[-(history - 1):])
        state.session_info['sessions'] = sessions
    sessions[session_id] = {'launched': dt.datetime.now().timestamp(), 'started': None, 'rendered': None, 'ended': None, 'user_agent': session_context.request.headers.get('User-Agent')}
    state.param.trigger('session_info')