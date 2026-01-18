from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
import param
from bokeh.models import CustomJS
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from bokeh.models.widgets.sliders import NumericalSlider as _BkNumericalSlider
from param.parameterized import resolve_value
from ..config import config
from ..io import state
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..util import (
from ..viewable import Layoutable
from ..widgets import FloatInput, IntInput
from .base import CompositeWidget, Widget
from .input import StaticText
class EditableIntSlider(_EditableContinuousSlider, IntSlider):
    """
    The EditableIntSlider widget allows selecting selecting an integer
    value within a set of bounds using a slider and for more precise
    control offers an editable integer input box.

    Reference: https://panel.holoviz.org/reference/widgets/EditableIntSlider.html

    :Example:

    >>> EditableIntSlider(
    ...     value=2, start=0, end=5, step=1, name="An integer value"
    ... )
    """
    fixed_start = param.Integer(default=None, doc='\n        A fixed lower bound for the slider and input.')
    fixed_end = param.Integer(default=None, doc='\n       A fixed upper bound for the slider and input.')
    _slider_widget: ClassVar[Type[Widget]] = IntSlider
    _input_widget: ClassVar[Type[Widget]] = IntInput