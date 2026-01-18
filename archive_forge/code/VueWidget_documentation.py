from traitlets import Unicode, Instance, Union, List, Any, Dict
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget_layout import Layout
from ipywidgets.widgets.widget import widget_serialization, CallbackDispatcher
from ipywidgets.widgets.trait_types import InstanceDict
from ._version import semver
from .ForceLoad import force_load_instance
Make the widget invisible