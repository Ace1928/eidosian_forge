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
def _load_entry_points(self):
    """
        Load entry points from external packages.
        Import is performed here, so any importlib
        can be easily bypassed by switching off the configuration flag.
        Also, there is no reason to waste time importing this module
        if it won't be used.
        """
    from .entry_points import load_entry_points
    load_entry_points('panel.extension')