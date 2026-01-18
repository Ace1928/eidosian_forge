import functools
import os
import sys
import collections
import importlib
import warnings
from contextvars import copy_context
from importlib.machinery import ModuleSpec
import pkgutil
import threading
import re
import logging
import time
import mimetypes
import hashlib
import base64
import traceback
from urllib.parse import urlparse
from typing import Dict, Optional, Union
import flask
from importlib_metadata import version as _get_distribution_version
from dash import dcc
from dash import html
from dash import dash_table
from .fingerprint import build_fingerprint, check_fingerprint
from .resources import Scripts, Css
from .dependencies import (
from .development.base_component import ComponentRegistry
from .exceptions import (
from .version import __version__
from ._configs import get_combined_config, pathname_configs, pages_folder_config
from ._utils import (
from . import _callback
from . import _get_paths
from . import _dash_renderer
from . import _validate
from . import _watch
from . import _get_app
from ._grouping import map_grouping, grouping_len, update_args_group
from . import _pages
from ._pages import (
from ._jupyter import jupyter_dash, JupyterDisplayMode
from .types import RendererHooks
def _on_assets_change(self, filename, modified, deleted):
    _reload = self._hot_reload
    with _reload.lock:
        _reload.hard = True
        _reload.hash = generate_hash()
        if self.config.assets_folder in filename:
            asset_path = os.path.relpath(filename, os.path.commonprefix([self.config.assets_folder, filename])).replace('\\', '/').lstrip('/')
            _reload.changed_assets.append({'url': self.get_asset_url(asset_path), 'modified': int(modified), 'is_css': filename.endswith('css')})
            if filename not in self._assets_files and (not deleted):
                res = self._add_assets_resource(asset_path, filename)
                if filename.endswith('js'):
                    self.scripts.append_script(res)
                elif filename.endswith('css'):
                    self.css.append_css(res)
            if deleted:
                if filename in self._assets_files:
                    self._assets_files.remove(filename)

                def delete_resource(resources):
                    to_delete = None
                    for r in resources:
                        if r.get('asset_path') == asset_path:
                            to_delete = r
                            break
                    if to_delete:
                        resources.remove(to_delete)
                if filename.endswith('js'):
                    delete_resource(self.scripts._resources._resources)
                elif filename.endswith('css'):
                    delete_resource(self.css._resources._resources)