import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _VuetifyTemplate(components: dict=None, css: str=None, data: str=None, events: list=[], layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, methods: str=None, tabbable: bool=None, template: typing.Union[Element[ipyvue.Template], str]=None, tooltip: str=None) -> Element[ipyvuetify.VuetifyTemplate]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...