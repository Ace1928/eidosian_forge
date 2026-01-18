from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
class sampledata_list(nodes.General, nodes.Element):

    def __init__(self, *args, **kwargs):
        self.sampledata_key = kwargs.pop('sampledata_key')
        super().__init__(*args, **kwargs)