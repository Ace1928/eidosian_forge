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
class NotebookBaseHandler(ExtensionHandlerJinjaMixin, ExtensionHandlerMixin, JupyterHandler):
    """The base notebook API handler."""

    @property
    def custom_css(self) -> t.Any:
        return self.settings.get('custom_css', True)

    def get_page_config(self) -> dict[str, t.Any]:
        """Get the page config."""
        config = LabConfig()
        app: JupyterNotebookApp = self.extensionapp
        base_url = self.settings.get('base_url', '/')
        page_config_data = self.settings.setdefault('page_config_data', {})
        page_config = {**page_config_data, 'appVersion': version, 'baseUrl': self.base_url, 'terminalsAvailable': self.settings.get('terminals_available', False), 'token': self.settings['token'], 'fullStaticUrl': ujoin(self.base_url, 'static', self.name), 'frontendUrl': ujoin(self.base_url, '/'), 'exposeAppInBrowser': app.expose_app_in_browser}
        server_root = self.settings.get('server_root_dir', '')
        server_root = server_root.replace(os.sep, '/')
        server_root = os.path.normpath(Path(server_root).expanduser())
        try:
            if self.serverapp.preferred_dir != server_root:
                page_config['preferredPath'] = '/' + os.path.relpath(self.serverapp.preferred_dir, server_root)
            else:
                page_config['preferredPath'] = '/'
        except Exception:
            page_config['preferredPath'] = '/'
        mathjax_config = self.settings.get('mathjax_config', 'TeX-AMS_HTML-full,Safe')
        mathjax_url = self.settings.get('mathjax_url', 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js')
        if not url_is_absolute(mathjax_url) and (not mathjax_url.startswith(self.base_url)):
            mathjax_url = ujoin(self.base_url, mathjax_url)
        page_config.setdefault('mathjaxConfig', mathjax_config)
        page_config.setdefault('fullMathjaxUrl', mathjax_url)
        page_config.setdefault('jupyterConfigDir', jupyter_config_dir())
        for name in config.trait_names():
            page_config[_camelCase(name)] = getattr(app, name)
        for name in config.trait_names():
            if not name.endswith('_url'):
                continue
            full_name = _camelCase('full_' + name)
            full_url = getattr(app, name)
            if not is_url(full_url):
                full_url = ujoin(base_url, full_url)
            page_config[full_name] = full_url
        labextensions_path = app.extra_labextensions_path + app.labextensions_path
        recursive_update(page_config, get_page_config(labextensions_path, logger=self.log))
        return page_config