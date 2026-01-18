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
def _setup_dev_tools(self, **kwargs):
    debug = kwargs.get('debug', False)
    dev_tools = self._dev_tools = AttributeDict()
    for attr in ('ui', 'props_check', 'serve_dev_bundles', 'hot_reload', 'silence_routes_logging', 'prune_errors'):
        dev_tools[attr] = get_combined_config(attr, kwargs.get(attr, None), default=debug)
    for attr, _type, default in (('hot_reload_interval', float, 3), ('hot_reload_watch_interval', float, 0.5), ('hot_reload_max_retry', int, 8)):
        dev_tools[attr] = _type(get_combined_config(attr, kwargs.get(attr, None), default=default))
    return dev_tools