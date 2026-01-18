from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
class _BoundedLogFloat(_Float):
    max = CFloat(4.0, help='Max value for the exponent').tag(sync=True)
    min = CFloat(0.0, help='Min value for the exponent').tag(sync=True)
    base = CFloat(10.0, help='Base of value').tag(sync=True)
    value = CFloat(1.0, help='Float value').tag(sync=True)

    @validate('value')
    def _validate_value(self, proposal):
        """Cap and floor value"""
        value = proposal['value']
        if self.base ** self.min > value or self.base ** self.max < value:
            value = min(max(value, self.base ** self.min), self.base ** self.max)
        return value

    @validate('min')
    def _validate_min(self, proposal):
        """Enforce base ** min <= value <= base ** max"""
        min = proposal['value']
        if min > self.max:
            raise TraitError('Setting min > max')
        if self.base ** min > self.value:
            self.value = self.base ** min
        return min

    @validate('max')
    def _validate_max(self, proposal):
        """Enforce base ** min <= value <= base ** max"""
        max = proposal['value']
        if max < self.min:
            raise TraitError('setting max < min')
        if self.base ** max < self.value:
            self.value = self.base ** max
        return max