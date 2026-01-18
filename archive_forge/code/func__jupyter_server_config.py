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
def _jupyter_server_config(self):
    """The jupyter server config."""
    base_config = {'ServerApp': {'default_url': self.default_url, 'open_browser': self.open_browser, 'file_url_prefix': self.file_url_prefix}}
    base_config['ServerApp'].update(self.serverapp_config)
    return base_config