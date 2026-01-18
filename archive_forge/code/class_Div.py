from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import Bool, String
from .widget import Widget
class Div(Markup):
    """ A block (div) of text.

    This Bokeh model corresponds to an HTML ``<div>`` element.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/interaction/widgets/div.py'
    render_as_text = Bool(False, help='\n    Whether the contents should be rendered as raw text or as interpreted HTML.\n    The default value is False, meaning contents are rendered as HTML.\n    ')