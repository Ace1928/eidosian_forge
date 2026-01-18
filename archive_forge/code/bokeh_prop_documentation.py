from __future__ import annotations
import logging  # isort:skip
import importlib
import textwrap
import warnings
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from bokeh.core.property._sphinx import type_link
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import PROP_DETAIL
 Required Sphinx extension setup function. 