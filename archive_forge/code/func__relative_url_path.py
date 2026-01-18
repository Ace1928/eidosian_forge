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
def _relative_url_path(relative_package_path='', namespace=''):
    if any((relative_package_path.startswith(x + '/') for x in ['dcc', 'html', 'dash_table'])):
        relative_package_path = relative_package_path.replace('dash.', '')
        version = importlib.import_module(f'{namespace}.{os.path.split(relative_package_path)[0]}').__version__
    else:
        version = importlib.import_module(namespace).__version__
    module_path = os.path.join(os.path.dirname(sys.modules[namespace].__file__), relative_package_path)
    modified = int(os.stat(module_path).st_mtime)
    fingerprint = build_fingerprint(relative_package_path, version, modified)
    return f'{self.config.requests_pathname_prefix}_dash-component-suites/{namespace}/{fingerprint}'