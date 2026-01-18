from __future__ import annotations
from collections.abc import Iterable, Mapping
from inspect import Parameter
from numbers import Integral, Number, Real
from typing import Any, Optional, Tuple
import param
from .base import Widget
from .input import Checkbox, TextInput
from .select import Select
from .slider import DiscreteSlider, FloatSlider, IntSlider
class fixed(param.Parameterized):
    """
    A pseudo-widget whose value is fixed and never synced to the client.
    """
    description = param.String(default='')
    value = param.Parameter(doc='Any Python object')

    def __init__(self, value: Any, **kwargs: Any):
        super().__init__(value=value, **kwargs)

    def get_interact_value(self):
        """
        Return the value for this widget which should be passed to
        interactive functions. Custom widgets can change this method
        to process the raw value ``self.value``.
        """
        return self.value