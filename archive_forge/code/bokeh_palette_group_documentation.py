from __future__ import annotations
import logging  # isort:skip
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.errors import SphinxError
import bokeh.palettes as bp
from . import PARALLEL_SAFE
from .templates import PALETTE_GROUP_DETAIL
 Required Sphinx extension setup function. 