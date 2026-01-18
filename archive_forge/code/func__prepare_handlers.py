from __future__ import annotations
import logging
import re
import sys
import typing as t
from jinja2 import Environment, FileSystemLoader
from jupyter_core.application import JupyterApp, NoStart
from tornado.log import LogFormatter
from tornado.web import RedirectHandler
from traitlets import Any, Bool, Dict, HasTraits, List, Unicode, default
from traitlets.config import Config
from jupyter_server.serverapp import ServerApp
from jupyter_server.transutils import _i18n
from jupyter_server.utils import is_namespace_package, url_path_join
from .handler import ExtensionHandlerMixin
def _prepare_handlers(self):
    """Prepare the handlers."""
    assert self.serverapp is not None
    webapp = self.serverapp.web_app
    self.initialize_handlers()
    new_handlers = []
    for handler_items in self.handlers:
        pattern = url_path_join(webapp.settings['base_url'], handler_items[0])
        handler = handler_items[1]
        kwargs: dict[str, t.Any] = {}
        if issubclass(handler, ExtensionHandlerMixin):
            kwargs['name'] = self.name
        try:
            kwargs.update(handler_items[2])
        except IndexError:
            pass
        new_handler = (pattern, handler, kwargs)
        new_handlers.append(new_handler)
    if len(self.static_paths) > 0:
        static_url = url_path_join(self.static_url_prefix, '(.*)')
        handler = (static_url, webapp.settings['static_handler_class'], {'path': self.static_paths})
        new_handlers.append(handler)
    webapp.add_handlers('.*$', new_handlers)