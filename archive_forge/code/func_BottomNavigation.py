import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
@implements(_BottomNavigation)
def BottomNavigation(**kwargs):
    if isinstance(kwargs.get('layout'), dict):
        kwargs['layout'] = w.Layout(**kwargs['layout'])
    widget_cls = ipyvuetify.generated.BottomNavigation
    comp = reacton.core.ComponentWidget(widget=widget_cls)
    return ValueElement('v_model', comp, kwargs=kwargs)