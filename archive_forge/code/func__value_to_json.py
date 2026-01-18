import os
from traitlets import Any, Unicode, List, Dict, Union, Instance
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget import widget_serialization
from .Template import Template, get_template
from ._version import semver
from .ForceLoad import force_load_instance
import inspect
from importlib import import_module
def _value_to_json(x, obj):
    if inspect.isclass(x):
        return {'class': [x.__module__, x.__name__], 'props': x.class_trait_names()}
    return widget_serialization['to_json'](x, obj)