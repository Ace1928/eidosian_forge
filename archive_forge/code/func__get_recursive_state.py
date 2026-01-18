import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
def _get_recursive_state(widget, store=None, drop_defaults=False):
    """Gets the embed state of a widget, and all other widgets it refers to as well"""
    if store is None:
        store = dict()
    state = widget._get_embed_state(drop_defaults=drop_defaults)
    store[widget.model_id] = state
    for ref in _find_widget_refs_by_state(widget, state['state']):
        if ref.model_id not in store:
            _get_recursive_state(ref, store, drop_defaults=drop_defaults)
    return store