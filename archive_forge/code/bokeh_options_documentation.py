from __future__ import annotations
import logging  # isort:skip
import importlib
import textwrap
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from bokeh.core.property._sphinx import type_link
from bokeh.util.options import Options
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective, py_sig_re
from .templates import OPTIONS_DETAIL
 Required Sphinx extension setup function. 