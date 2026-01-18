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
class TreeHandler(NotebookBaseHandler):
    """A tree page handler."""

    @web.authenticated
    async def get(self, path: str='') -> None:
        """
        Display appropriate page for given path.

        - A directory listing is shown if path is a directory
        - Redirected to notebook page if path is a notebook
        - Render the raw file if path is any other file
        """
        path = path.strip('/')
        cm = self.contents_manager
        if await ensure_async(cm.dir_exists(path=path)):
            if await ensure_async(cm.is_hidden(path)) and (not cm.allow_hidden):
                self.log.info('Refusing to serve hidden directory, via 404 Error')
                raise web.HTTPError(404)
            page_config = self.get_page_config()
            page_config['treePath'] = path
            tpl = self.render_template('tree.html', page_config=page_config)
            return self.write(tpl)
        if await ensure_async(cm.file_exists(path)):
            model = await ensure_async(cm.get(path, content=False))
            if model['type'] == 'notebook':
                url = ujoin(self.base_url, 'notebooks', url_escape(path))
            else:
                url = ujoin(self.base_url, 'files', url_escape(path))
            self.log.debug('Redirecting %s to %s', self.request.path, url)
            self.redirect(url)
            return None
        raise web.HTTPError(404)