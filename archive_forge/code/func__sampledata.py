from __future__ import annotations
from sphinx.util import logging  # isort:skip
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import EXAMPLE_METADATA
from .util import get_sphinx_resources
def _sampledata(mods: str | None) -> str | None:
    if mods is None:
        return
    mods = mods.split('#')[0].strip()
    mods = (mod.strip() for mod in mods.split(','))
    return ', '.join((f':ref:`bokeh.sampledata.{mod} <sampledata_{mod}>`' for mod in mods))