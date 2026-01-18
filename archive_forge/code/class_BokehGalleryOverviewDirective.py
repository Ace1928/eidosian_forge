from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
class BokehGalleryOverviewDirective(BokehDirective):
    has_content = False
    required_arguments = 1

    def run(self):
        return [gallery_xrefs('', subfolder=self.arguments[0])]