from a list of options.
from __future__ import annotations
import itertools
import re
from types import FunctionType
from typing import (
import numpy as np
import param
from bokeh.models import PaletteSelect
from bokeh.models.widgets import (
from ..io.resources import CDN_DIST
from ..layout.base import Column, ListPanel, NamedListPanel
from ..models import (
from ..util import PARAM_NAME_PATTERN, indexOf, isIn
from ._mixin import TooltipMixin
from .base import CompositeWidget, Widget
from .button import Button, _ButtonBase
from .input import TextAreaInput, TextInput
class CheckBoxGroup(_CheckGroupBase):
    """
    The `CheckBoxGroup` widget allows selecting between a list of options by
    ticking the corresponding checkboxes.

    It falls into the broad category of multi-option selection widgets that
    provide a compatible API and include the `MultiSelect`, `CrossSelector`
    and `CheckButtonGroup` widgets.

    Reference: https://panel.holoviz.org/reference/widgets/CheckBoxGroup.html

    :Example:

    >>> CheckBoxGroup(
    ...     name='Fruits', value=['Apple', 'Pear'], options=['Apple', 'Banana', 'Pear', 'Strawberry'],
    ...     inline=True
    ... )
    """
    inline = param.Boolean(default=False, doc='\n        Whether the items be arrange vertically (``False``) or\n        horizontally in-line (``True``).')
    _widget_type: ClassVar[Type[Model]] = _BkCheckboxGroup