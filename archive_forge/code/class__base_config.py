import ast
import copy
import importlib
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from weakref import WeakKeyDictionary
import param
from bokeh.core.has_props import _default_resolver
from bokeh.document import Document
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from pyviz_comms import (
from .io.logging import panel_log_handler
from .io.state import state
from .util import param_watchers
class _base_config(param.Parameterized):
    css_files = param.List(default=[], doc='\n        External CSS files to load.')
    js_files = param.Dict(default={}, doc='\n        External JS files to load. Dictionary should map from exported\n        name to the URL of the JS file.')
    js_modules = param.Dict(default={}, doc='\n        External JS files to load as modules. Dictionary should map from\n        exported name to the URL of the JS file.')
    raw_css = param.List(default=[], doc='\n        List of raw CSS strings to add to load.')