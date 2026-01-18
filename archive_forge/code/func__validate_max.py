from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget import register
from .widget_core import CoreWidget
from .trait_types import Date, date_serialization
from traitlets import Unicode, Bool, Union, CInt, CaselessStrEnum, TraitError, validate
@validate('max')
def _validate_max(self, proposal):
    """Enforce min <= value <= max"""
    max = proposal['value']
    if max is None:
        return max
    if self.min and max < self.min:
        raise TraitError('setting max < min')
    if self.value and max < self.value:
        self.value = max
    return max