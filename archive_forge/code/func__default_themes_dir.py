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
@default('themes_dir')
def _default_themes_dir(self) -> str:
    return str(app_dir / 'themes')