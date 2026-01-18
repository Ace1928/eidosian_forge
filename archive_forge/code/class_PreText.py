from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import Bool, String
from .widget import Widget
class PreText(Paragraph):
    """ A block (paragraph) of pre-formatted text.

    This Bokeh model corresponds to an HTML ``<pre>`` element.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/interaction/widgets/pretext.py'