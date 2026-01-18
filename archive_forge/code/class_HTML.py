from __future__ import annotations
import logging # isort:skip
from typing import Any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import NOT_A_PROPERTY_OF
from ..model import Model, Qualified
from .css import Styles
from .ui.ui_element import UIElement
class HTML(DOMNode):
    """ A parsed HTML fragment with optional references to DOM nodes and UI elements. """

    def __init__(self, *html: str | DOMNode | UIElement, **kwargs: Any) -> None:
        if html and 'html' in kwargs:
            raise TypeError("'html' argument specified multiple times")
        processed_html: Init[str | list[str | DOMNode | UIElement]]
        if not html:
            processed_html = kwargs.pop('html', Intrinsic)
        else:
            processed_html = list(html)
        super().__init__(html=processed_html, **kwargs)
    html = Required(Either(String, List(Either(String, Instance(DOMNode), Instance(UIElement)))), help='\n    Either a parsed HTML string with optional references to Bokeh objects using\n    ``<ref id="..."></ref>`` syntax. Or a list of parsed HTML interleaved with\n    Bokeh\'s objects. Any DOM node or UI element (even a plot) can be referenced\n    here.\n    ')
    refs = List(Either(String, Instance(DOMNode), Instance(UIElement)), default=[], help='\n    A collection of objects referenced by ``<ref id="..."></ref>`` from `the `html`` property.\n    Objects already included by instance in ``html`` don\'t have to be repeated here.\n    ')