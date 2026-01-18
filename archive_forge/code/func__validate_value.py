from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget import register
from .widget_core import CoreWidget
from .trait_types import Date, date_serialization
from traitlets import Unicode, Bool, Union, CInt, CaselessStrEnum, TraitError, validate
@validate('value')
def _validate_value(self, proposal):
    """Cap and floor value"""
    value = proposal['value']
    if value is None:
        return value
    if self.min and self.min > value:
        value = max(value, self.min)
    if self.max and self.max < value:
        value = min(value, self.max)
    return value