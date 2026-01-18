from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
class _BoundedFloatRange(_FloatRange):
    step = CFloat(1.0, help='Minimum step that the value can take (ignored by some views)').tag(sync=True)
    max = CFloat(100.0, help='Max value').tag(sync=True)
    min = CFloat(0.0, help='Min value').tag(sync=True)

    def __init__(self, *args, **kwargs):
        min, max = (kwargs.get('min', 0.0), kwargs.get('max', 100.0))
        if kwargs.get('value', None) is None:
            kwargs['value'] = (0.75 * min + 0.25 * max, 0.25 * min + 0.75 * max)
        elif not isinstance(kwargs['value'], tuple):
            try:
                kwargs['value'] = tuple(kwargs['value'])
            except:
                raise TypeError("A 'range' must be able to be cast to a tuple. The input of type {} could not be cast to a tuple".format(type(kwargs['value'])))
        super().__init__(*args, **kwargs)

    @validate('min', 'max')
    def _validate_bounds(self, proposal):
        trait = proposal['trait']
        new = proposal['value']
        if trait.name == 'min' and new > self.max:
            raise TraitError('setting min > max')
        if trait.name == 'max' and new < self.min:
            raise TraitError('setting max < min')
        if trait.name == 'min':
            self.value = (max(new, self.value[0]), max(new, self.value[1]))
        if trait.name == 'max':
            self.value = (min(new, self.value[0]), min(new, self.value[1]))
        return new

    @validate('value')
    def _validate_value(self, proposal):
        lower, upper = super()._validate_value(proposal)
        lower, upper = (min(lower, self.max), min(upper, self.max))
        lower, upper = (max(lower, self.min), max(upper, self.min))
        return (lower, upper)