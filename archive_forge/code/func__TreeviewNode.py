import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _TreeviewNode(activatable: bool=None, active_class: str=None, attributes: dict={}, children: list=[], class_: str=None, color: str=None, expand_icon: str=None, indeterminate_icon: str=None, item: dict=None, item_children: str=None, item_disabled: str=None, item_key: str=None, item_text: str=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, level: float=None, loading_icon: str=None, off_icon: str=None, on_icon: str=None, open_on_click: bool=None, rounded: bool=None, selectable: bool=None, selected_color: str=None, shaped: bool=None, slot: str=None, style_: str=None, tabbable: bool=None, tooltip: str=None, transition: bool=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.TreeviewNode, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...