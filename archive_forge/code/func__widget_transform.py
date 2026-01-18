from __future__ import annotations
import math
from typing import (
import param  # type: ignore
from bokeh.models import ImportedStyleSheet, Tooltip
from bokeh.models.dom import HTML
from param.parameterized import register_reference_transform
from .._param import Margin
from ..layout.base import Row
from ..reactive import Reactive
from ..viewable import Layoutable, Viewable
def _widget_transform(obj):
    return obj.param.value if isinstance(obj, Widget) else obj