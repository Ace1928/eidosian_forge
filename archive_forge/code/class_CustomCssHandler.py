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
class CustomCssHandler(NotebookBaseHandler):
    """A custom CSS handler."""

    @web.authenticated
    def get(self) -> t.Any:
        """Get the custom css file."""
        self.set_header('Content-Type', 'text/css')
        page_config = self.get_page_config()
        custom_css_file = f'{page_config['jupyterConfigDir']}/custom/custom.css'
        if not Path(custom_css_file).is_file():
            static_path_root = re.match('^(.*?)static', page_config['staticDir'])
            if static_path_root is not None:
                custom_dir = static_path_root.groups()[0]
                custom_css_file = f'{custom_dir}custom/custom.css'
        with Path(custom_css_file).open() as css_f:
            return self.write(css_f.read())