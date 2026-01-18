from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
class BokehSampledataXrefDirective(BokehDirective):
    has_content = False
    required_arguments = 1

    def run(self):
        return [sampledata_list('', sampledata_key=self.arguments[0])]