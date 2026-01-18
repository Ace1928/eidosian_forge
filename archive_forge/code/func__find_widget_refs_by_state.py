import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
def _find_widget_refs_by_state(widget, state):
    """Find references to other widgets in a widget's state"""
    keys = tuple(state.keys())
    for key in keys:
        value = getattr(widget, key)
        if isinstance(value, Widget):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Widget):
                    yield item
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, Widget):
                    yield item