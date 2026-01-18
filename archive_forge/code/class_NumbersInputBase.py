from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget import register
from .trait_types import Color, NumberFormat
class NumbersInputBase(TagsInput):
    _model_name = Unicode('NumbersInputBaseModel').tag(sync=True)
    min = CFloat(default_value=None, allow_none=True).tag(sync=True)
    max = CFloat(default_value=None, allow_none=True).tag(sync=True)

    @validate('value')
    def _validate_numbers(self, proposal):
        for tag_value in proposal['value']:
            if self.min is not None and tag_value < self.min:
                raise TraitError('Tag value {} should be >= {}'.format(tag_value, self.min))
            if self.max is not None and tag_value > self.max:
                raise TraitError('Tag value {} should be <= {}'.format(tag_value, self.max))
        return proposal['value']