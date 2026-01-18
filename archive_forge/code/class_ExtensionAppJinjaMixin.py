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
class ExtensionAppJinjaMixin(HasTraits):
    """Use Jinja templates for HTML templates on top of an ExtensionApp."""
    jinja2_options = Dict(help=_i18n('Options to pass to the jinja2 environment for this\n        ')).tag(config=True)

    @t.no_type_check
    def _prepare_templates(self):
        """Get templates defined in a subclass."""
        self.initialize_templates()
        if len(self.template_paths) > 0:
            self.settings.update({f'{self.name}_template_paths': self.template_paths})
        self.jinja2_env = Environment(loader=FileSystemLoader(self.template_paths), extensions=['jinja2.ext.i18n'], autoescape=True, **self.jinja2_options)
        self.settings.update({f'{self.name}_jinja2_env': self.jinja2_env})