import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _Data(attributes: dict={}, children: list=[], class_: str=None, disable_filtering: bool=None, disable_pagination: bool=None, disable_sort: bool=None, group_by: typing.Union[str, list]=None, group_desc: typing.Union[bool, list]=None, items: list=[], items_per_page: float=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, locale: str=None, multi_sort: bool=None, must_sort: bool=None, options: dict=None, page: float=None, search: str=None, server_items_length: float=None, slot: str=None, sort_by: typing.Union[str, list]=None, sort_desc: typing.Union[bool, list]=None, style_: str=None, tabbable: bool=None, tooltip: str=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.Data, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...