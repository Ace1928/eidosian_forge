from __future__ import annotations
import logging # isort:skip
import math
from collections import defaultdict
from typing import (
from .core.enums import Location, LocationType, SizingModeType
from .core.property.singletons import Undefined, UndefinedType
from .models import (
from .util.dataclasses import dataclass
from .util.warnings import warn
def _handle_child_sizing(children: list[UIElement], sizing_mode: SizingModeType | None, *, widget: str) -> None:
    for item in children:
        if isinstance(item, UIElement):
            continue
        if not isinstance(item, LayoutDOM):
            raise ValueError(f'Only LayoutDOM items can be inserted into a {widget}. Tried to insert: {item} of type {type(item)}')
        if sizing_mode is not None and _has_auto_sizing(item):
            item.sizing_mode = sizing_mode