from __future__ import annotations
import os
import pathlib
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from jupyter_server.base.handlers import FileFindHandler, JupyterHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import template, web
from .config import LabConfig, get_page_config, recursive_update
from .licenses_handler import LicensesHandler, LicensesManager
from .listings_handler import ListingsHandler, fetch_listings
from .settings_handler import SettingsHandler
from .themes_handler import ThemesHandler
from .translations_handler import TranslationsHandler
from .workspaces_handler import WorkspacesHandler, WorkspacesManager
class LabHandler(ExtensionHandlerJinjaMixin, ExtensionHandlerMixin, JupyterHandler):
    """Render the JupyterLab View."""

    @lru_cache
    def get_page_config(self) -> dict[str, Any]:
        """Construct the page config object"""
        self.application.store_id = getattr(self.application, 'store_id', 0)
        config = LabConfig()
        app: LabServerApp = self.extensionapp
        settings_dir = app.app_settings_dir
        page_config = self.settings.setdefault('page_config_data', {})
        terminals = self.settings.get('terminals_available', False)
        server_root = self.settings.get('server_root_dir', '')
        server_root = server_root.replace(os.sep, '/')
        base_url = self.settings.get('base_url')
        full_static_url = self.static_url_prefix.rstrip('/')
        page_config.setdefault('fullStaticUrl', full_static_url)
        page_config.setdefault('terminalsAvailable', terminals)
        page_config.setdefault('ignorePlugins', [])
        page_config.setdefault('serverRoot', server_root)
        page_config['store_id'] = self.application.store_id
        server_root = os.path.normpath(os.path.expanduser(server_root))
        preferred_path = ''
        try:
            preferred_path = self.serverapp.contents_manager.preferred_dir
        except Exception:
            try:
                if self.serverapp.preferred_dir and self.serverapp.preferred_dir != server_root:
                    preferred_path = pathlib.Path(self.serverapp.preferred_dir).relative_to(server_root).as_posix()
            except Exception:
                pass
        page_config['preferredPath'] = preferred_path or '/'
        self.application.store_id += 1
        mathjax_config = self.settings.get('mathjax_config', 'TeX-AMS_HTML-full,Safe')
        mathjax_url = self.mathjax_url
        if not mathjax_url:
            mathjax_url = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js'
        page_config.setdefault('mathjaxConfig', mathjax_config)
        page_config.setdefault('fullMathjaxUrl', mathjax_url)
        for name in config.trait_names():
            page_config[_camelCase(name)] = getattr(app, name)
        for name in config.trait_names():
            if not name.endswith('_url'):
                continue
            full_name = _camelCase('full_' + name)
            full_url = getattr(app, name)
            if base_url is not None and (not is_url(full_url)):
                full_url = ujoin(base_url, full_url)
            page_config[full_name] = full_url
        labextensions_path = app.extra_labextensions_path + app.labextensions_path
        recursive_update(page_config, get_page_config(labextensions_path, settings_dir, logger=self.log))
        page_config_hook = self.settings.get('page_config_hook', None)
        if page_config_hook:
            page_config = page_config_hook(self, page_config)
        return page_config

    @web.authenticated
    @web.removeslash
    def get(self, mode: str | None=None, workspace: str | None=None, tree: str | None=None) -> None:
        """Get the JupyterLab html page."""
        workspace = 'default' if workspace is None else workspace.replace('/workspaces/', '')
        tree_path = '' if tree is None else tree.replace('/tree/', '')
        page_config = self.get_page_config()
        if mode == 'doc':
            page_config['mode'] = 'single-document'
        else:
            page_config['mode'] = 'multiple-document'
        page_config['workspace'] = workspace
        page_config['treePath'] = tree_path
        tpl = self.render_template('index.html', page_config=page_config)
        self.write(tpl)