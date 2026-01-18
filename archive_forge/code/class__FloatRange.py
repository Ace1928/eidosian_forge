from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
class _FloatRange(_Float):
    value = Tuple(CFloat(), CFloat(), default_value=(0.0, 1.0), help='Tuple of (lower, upper) bounds').tag(sync=True)

    @property
    def lower(self):
        return self.value[0]

    @lower.setter
    def lower(self, lower):
        self.value = (lower, self.value[1])

    @property
    def upper(self):
        return self.value[1]

    @upper.setter
    def upper(self, upper):
        self.value = (self.value[0], upper)

    @validate('value')
    def _validate_value(self, proposal):
        lower, upper = proposal['value']
        if upper < lower:
            raise TraitError('setting lower > upper')
        return (lower, upper)