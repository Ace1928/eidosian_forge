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
class ComponentResourceHandler(StaticFileHandler):
    """
    A handler that serves local resources relative to a Python module.
    The handler resolves a specific Panel component by module reference
    and name, then resolves an attribute on that component to check
    if it contains the requested resource path.

    /<endpoint>/<module>/<class>/<attribute>/<path>
    """
    _resource_attrs = ['__css__', '__javascript__', '__js_module__', '__javascript_modules__', '_resources', '_css', '_js', 'base_css', 'css', '_stylesheets', 'modifiers']

    def initialize(self, path: Optional[str]=None, default_filename: Optional[str]=None):
        self.root = path
        self.default_filename = default_filename

    def parse_url_path(self, path: str) -> str:
        """
        Resolves the resource the URL pattern refers to.
        """
        parts = path.split('/')
        if len(parts) < 4:
            raise HTTPError(400, 'Malformed URL')
        mod, cls, rtype, *subpath = parts
        try:
            module = importlib.import_module(mod)
        except ModuleNotFoundError:
            raise HTTPError(404, 'Module not found') from None
        try:
            component = getattr(module, cls)
        except AttributeError:
            raise HTTPError(404, 'Component not found') from None
        if rtype not in self._resource_attrs:
            raise HTTPError(403, 'Requested resource type not valid.')
        try:
            resources = getattr(component, rtype)
        except AttributeError:
            raise HTTPError(404, 'Resource type not found') from None
        if rtype == '_resources':
            rtype = subpath[0]
            subpath = subpath[1:]
            if rtype not in resources:
                raise HTTPError(404, 'Resource type not found')
            resources = resources[rtype]
            rtype = f'_resources/{rtype}'
        elif rtype == 'modifiers':
            resources = [st for rs in resources.values() for st in rs.get('stylesheets', []) if isinstance(st, str)]
        if isinstance(resources, dict):
            resources = list(resources.values())
        elif isinstance(resources, (str, pathlib.PurePath)):
            resources = [resources]
        resources = [str(resolve_custom_path(component, resource, relative=True)).replace(os.path.sep, '/') for resource in resources]
        rel_path = '/'.join(subpath)
        if rel_path not in resources:
            raise HTTPError(403, 'Requested resource was not listed.')
        if not module.__file__:
            raise HTTPError(404, 'Requested module does not reference a file.')
        return str(pathlib.Path(module.__file__).parent / rel_path)

    @classmethod
    def get_absolute_path(cls, root: str, path: str) -> str:
        return path

    def validate_absolute_path(self, root: str, absolute_path: str) -> str:
        if not os.path.exists(absolute_path):
            raise HTTPError(404)
        if not os.path.isfile(absolute_path):
            raise HTTPError(403, '%s is not a file', self.path)
        return absolute_path