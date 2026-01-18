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
class RadioBoxGroup(_RadioGroupBase):
    """
    The `RadioBoxGroup` widget allows selecting from a list or dictionary of
    values using a set of checkboxes.

    It falls into the broad category of single-value, option-selection widgets
    that provide a compatible API and include the `RadioButtonGroup`, `Select`
    and `DiscreteSlider` widgets.

    Reference: https://panel.holoviz.org/reference/widgets/RadioBoxGroup.html

    :Example:

    >>> RadioBoxGroup(
    ...     name='Sponsor', options=['Anaconda', 'Blackstone'], inline=True
    ... )
    """
    inline = param.Boolean(default=False, doc='\n        Whether the items be arrange vertically (``False``) or\n        horizontally in-line (``True``).')
    _supports_embed: ClassVar[bool] = True
    _widget_type: ClassVar[Type[Model]] = _BkRadioBoxGroup