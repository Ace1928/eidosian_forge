from traitlets import Unicode, Instance, Union, List, Any, Dict
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget_layout import Layout
from ipywidgets.widgets.widget import widget_serialization, CallbackDispatcher
from ipywidgets.widgets.trait_types import InstanceDict
from ._version import semver
from .ForceLoad import force_load_instance
def _fire_event(self, event, data=None):
    dispatcher = self._event_handlers_map[event]
    for callback in dispatcher.callbacks:
        callback(self, event, data or {})