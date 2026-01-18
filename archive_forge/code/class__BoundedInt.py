from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
class _BoundedInt(_Int):
    """Base class for widgets that represent an integer bounded from above and below.
    """
    max = CInt(100, help='Max value').tag(sync=True)
    min = CInt(0, help='Min value').tag(sync=True)

    def __init__(self, value=None, min=None, max=None, step=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        if min is not None:
            kwargs['min'] = min
        if max is not None:
            kwargs['max'] = max
        if step is not None:
            kwargs['step'] = step
        super().__init__(**kwargs)

    @validate('value')
    def _validate_value(self, proposal):
        """Cap and floor value"""
        value = proposal['value']
        if self.min > value or self.max < value:
            value = min(max(value, self.min), self.max)
        return value

    @validate('min')
    def _validate_min(self, proposal):
        """Enforce min <= value <= max"""
        min = proposal['value']
        if min > self.max:
            raise TraitError('setting min > max')
        if min > self.value:
            self.value = min
        return min

    @validate('max')
    def _validate_max(self, proposal):
        """Enforce min <= value <= max"""
        max = proposal['value']
        if max < self.min:
            raise TraitError('setting max < min')
        if max < self.value:
            self.value = max
        return max