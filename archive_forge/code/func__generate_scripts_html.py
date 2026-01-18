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
def _generate_scripts_html(self):
    mode = 'dev' if self._dev_tools['props_check'] is True else 'prod'
    deps = [{key: value[mode] if isinstance(value, dict) else value for key, value in js_dist_dependency.items()} for js_dist_dependency in _dash_renderer._js_dist_dependencies]
    dev = self._dev_tools.serve_dev_bundles
    srcs = self._collect_and_register_resources(self.scripts._resources._filter_resources(deps, dev_bundles=dev)) + self.config.external_scripts + self._collect_and_register_resources(self.scripts.get_all_scripts(dev_bundles=dev) + self.scripts._resources._filter_resources(_dash_renderer._js_dist, dev_bundles=dev) + self.scripts._resources._filter_resources(dcc._js_dist, dev_bundles=dev) + self.scripts._resources._filter_resources(html._js_dist, dev_bundles=dev) + self.scripts._resources._filter_resources(dash_table._js_dist, dev_bundles=dev))
    self._inline_scripts.extend(_callback.GLOBAL_INLINE_SCRIPTS)
    _callback.GLOBAL_INLINE_SCRIPTS.clear()
    return '\n'.join([format_tag('script', src) if isinstance(src, dict) else f'<script src="{src}"></script>' for src in srcs] + [f'<script>{src}</script>' for src in self._inline_scripts])