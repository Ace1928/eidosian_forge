from __future__ import annotations
import logging  # isort:skip
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from bokeh.colors import named
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import COLOR_DETAIL
 Required Sphinx extension setup function. 