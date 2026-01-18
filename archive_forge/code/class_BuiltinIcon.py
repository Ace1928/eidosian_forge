from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.bases import Init
from ...core.property.singletons import Intrinsic
from .ui_element import UIElement
class BuiltinIcon(Icon):
    """ Built-in icons included with BokehJS. """

    def __init__(self, icon_name: Init[str]=Intrinsic, **kwargs) -> None:
        super().__init__(icon_name=icon_name, **kwargs)
    icon_name = Required(Either(Enum(ToolIcon), String), help='\n    The name of a built-in icon to use. Currently, the following icon names are\n    supported: ``"help"``, ``"question-mark"``, ``"settings"``, ``"x"``\n\n    .. bokeh-plot::\n        :source-position: none\n\n        from bokeh.io import show\n        from bokeh.layouts import column\n        from bokeh.models import BuiltinIcon, Button\n\n        builtin_icons = ["help", "question-mark", "settings", "x"]\n\n        icon_demo = []\n        for icon in builtin_icons:\n            icon_demo.append(Button(label=icon, button_type="light", icon=BuiltinIcon(icon, size="1.2em")))\n\n        show(column(icon_demo))\n\n    ')
    color = Color(default='gray', help='\n    Color to use for the icon.\n    ')