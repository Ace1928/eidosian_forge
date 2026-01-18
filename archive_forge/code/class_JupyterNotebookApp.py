from __future__ import annotations
import os
import re
import typing as t
from pathlib import Path
from jupyter_client.utils import ensure_async  # type:ignore[attr-defined]
from jupyter_core.application import base_aliases
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.extension.handler import (
from jupyter_server.serverapp import flags
from jupyter_server.utils import url_escape, url_is_absolute
from jupyter_server.utils import url_path_join as ujoin
from jupyterlab.commands import (  # type:ignore[import-untyped]
from jupyterlab_server import LabServerApp
from jupyterlab_server.config import (  # type:ignore[attr-defined]
from jupyterlab_server.handlers import _camelCase, is_url
from notebook_shim.shim import NotebookConfigShimMixin  # type:ignore[import-untyped]
from tornado import web
from traitlets import Bool, Unicode, default
from traitlets.config.loader import Config
from ._version import __version__
class JupyterNotebookApp(NotebookConfigShimMixin, LabServerApp):
    """The notebook server extension app."""
    name = 'notebook'
    app_name = 'Jupyter Notebook'
    description = 'Jupyter Notebook - A web-based notebook environment for interactive computing'
    version = version
    app_version = Unicode(version, help='The version of the application.')
    extension_url = '/'
    default_url = Unicode('/tree', config=True, help='The default URL to redirect to from `/`')
    file_url_prefix = '/tree'
    load_other_extensions = True
    app_dir = app_dir
    subcommands: dict[str, t.Any] = {}
    expose_app_in_browser = Bool(False, config=True, help='Whether to expose the global app instance to browser via window.jupyterapp')
    custom_css = Bool(True, config=True, help='Whether custom CSS is loaded on the page.\n        Defaults to True and custom CSS is loaded.\n        ')
    flags: Flags = flags
    flags['expose-app-in-browser'] = ({'JupyterNotebookApp': {'expose_app_in_browser': True}}, 'Expose the global app instance to browser via window.jupyterapp.')
    flags['custom-css'] = ({'JupyterNotebookApp': {'custom_css': True}}, 'Load custom CSS in template html files. Default is True')

    @default('static_dir')
    def _default_static_dir(self) -> str:
        return str(HERE / 'static')

    @default('templates_dir')
    def _default_templates_dir(self) -> str:
        return str(HERE / 'templates')

    @default('app_settings_dir')
    def _default_app_settings_dir(self) -> str:
        return str(app_dir / 'settings')

    @default('schemas_dir')
    def _default_schemas_dir(self) -> str:
        return str(app_dir / 'schemas')

    @default('themes_dir')
    def _default_themes_dir(self) -> str:
        return str(app_dir / 'themes')

    @default('user_settings_dir')
    def _default_user_settings_dir(self) -> str:
        return t.cast(str, get_user_settings_dir())

    @default('workspaces_dir')
    def _default_workspaces_dir(self) -> str:
        return t.cast(str, get_workspaces_dir())

    def _prepare_templates(self) -> None:
        super(LabServerApp, self)._prepare_templates()
        self.jinja2_env.globals.update(custom_css=self.custom_css)

    def server_extension_is_enabled(self, extension: str) -> bool:
        """Check if server extension is enabled."""
        if self.serverapp is None:
            return False
        try:
            extension_enabled = self.serverapp.extension_manager.extensions[extension].enabled is True
        except (AttributeError, KeyError, TypeError):
            extension_enabled = False
        return extension_enabled

    def initialize_handlers(self) -> None:
        """Initialize handlers."""
        assert self.serverapp is not None
        page_config = self.serverapp.web_app.settings.setdefault('page_config_data', {})
        nbclassic_enabled = self.server_extension_is_enabled('nbclassic')
        page_config['nbclassic_enabled'] = nbclassic_enabled
        if 'hub_prefix' in self.serverapp.tornado_settings:
            tornado_settings = self.serverapp.tornado_settings
            hub_prefix = tornado_settings['hub_prefix']
            page_config['hubPrefix'] = hub_prefix
            page_config['hubHost'] = tornado_settings['hub_host']
            page_config['hubUser'] = tornado_settings['user']
            page_config['shareUrl'] = ujoin(hub_prefix, 'user-redirect')
            if hasattr(self.serverapp, 'server_name'):
                page_config['hubServerName'] = self.serverapp.server_name
            page_config['token'] = ''
        self.handlers.append(('/tree(.*)', TreeHandler))
        self.handlers.append(('/notebooks(.*)', NotebookHandler))
        self.handlers.append(('/edit(.*)', FileHandler))
        self.handlers.append(('/consoles/(.*)', ConsoleHandler))
        self.handlers.append(('/terminals/(.*)', TerminalHandler))
        self.handlers.append(('/custom/custom.css', CustomCssHandler))
        super().initialize_handlers()

    def initialize(self, argv: list[str] | None=None) -> None:
        """Subclass because the ExtensionApp.initialize() method does not take arguments"""
        super().initialize()