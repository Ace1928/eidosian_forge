from __future__ import annotations
from sphinx.util import logging  # isort:skip
import re
import warnings
from os import getenv
from os.path import basename, dirname, join
from uuid import uuid4
from docutils import nodes
from docutils.parsers.rst.directives import choice, flag
from sphinx.errors import SphinxError
from sphinx.util import copyfile, ensuredir
from sphinx.util.display import status_iterator
from sphinx.util.nodes import set_source_info
from bokeh.document import Document
from bokeh.embed import autoload_static
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .example_handler import ExampleHandler
from .util import _REPO_TOP, get_sphinx_resources
def _remove_module_docstring(source, docstring):
    if docstring is None:
        return source
    return re.sub(f"""(\\'\\'\\'|\\"\\"\\")\\s*{re.escape(docstring)}\\s*(\\'\\'\\'|\\"\\"\\")""", '', source)