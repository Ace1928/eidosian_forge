from __future__ import annotations
import html
import json
import os
import typing as t
import uuid
import warnings
from pathlib import Path
from jinja2 import (
from jupyter_core.paths import jupyter_path
from nbformat import NotebookNode
from traitlets import Bool, Dict, HasTraits, List, Unicode, default, observe, validate
from traitlets.config import Config
from traitlets.utils.importstring import import_item
from nbconvert import filters
from .exporter import Exporter
def _register_filter(self, environ, name, jinja_filter):
    """
        Register a filter.
        A filter is a function that accepts and acts on one string.
        The filters are accessible within the Jinja templating engine.

        Parameters
        ----------
        name : str
            name to give the filter in the Jinja engine
        filter : filter
        """
    if jinja_filter is None:
        msg = 'filter'
        raise TypeError(msg)
    isclass = isinstance(jinja_filter, type)
    constructed = not isclass
    if constructed and isinstance(jinja_filter, (str,)):
        filter_cls = import_item(jinja_filter)
        return self._register_filter(environ, name, filter_cls)
    if constructed and callable(jinja_filter):
        environ.filters[name] = jinja_filter
        return jinja_filter
    if isclass and issubclass(jinja_filter, HasTraits):
        filter_instance = jinja_filter(parent=self)
        self._register_filter(environ, name, filter_instance)
        return None
    if isclass:
        filter_instance = jinja_filter()
        self._register_filter(environ, name, filter_instance)
        return None
    msg = 'filter'
    raise TypeError(msg)