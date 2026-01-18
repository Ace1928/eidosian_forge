import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _VuetifyWidget(attributes: dict={}, children: list=[], class_: str=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, slot: str=None, style_: str=None, tabbable: bool=None, tooltip: str=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.VuetifyWidget, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...