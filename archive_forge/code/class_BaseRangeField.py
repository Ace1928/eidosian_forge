from django import forms
from django.core import exceptions
from django.db.backends.postgresql.psycopg_any import (
from django.forms.widgets import HiddenInput, MultiWidget
from django.utils.translation import gettext_lazy as _
class BaseRangeField(forms.MultiValueField):
    default_error_messages = {'invalid': _('Enter two valid values.'), 'bound_ordering': _('The start of the range must not exceed the end of the range.')}
    hidden_widget = HiddenRangeWidget

    def __init__(self, **kwargs):
        if 'widget' not in kwargs:
            kwargs['widget'] = RangeWidget(self.base_field.widget)
        if 'fields' not in kwargs:
            kwargs['fields'] = [self.base_field(required=False), self.base_field(required=False)]
        kwargs.setdefault('required', False)
        kwargs.setdefault('require_all_fields', False)
        self.range_kwargs = {}
        if (default_bounds := kwargs.pop('default_bounds', None)):
            self.range_kwargs = {'bounds': default_bounds}
        super().__init__(**kwargs)

    def prepare_value(self, value):
        lower_base, upper_base = self.fields
        if isinstance(value, self.range_type):
            return [lower_base.prepare_value(value.lower), upper_base.prepare_value(value.upper)]
        if value is None:
            return [lower_base.prepare_value(None), upper_base.prepare_value(None)]
        return value

    def compress(self, values):
        if not values:
            return None
        lower, upper = values
        if lower is not None and upper is not None and (lower > upper):
            raise exceptions.ValidationError(self.error_messages['bound_ordering'], code='bound_ordering')
        try:
            range_value = self.range_type(lower, upper, **self.range_kwargs)
        except TypeError:
            raise exceptions.ValidationError(self.error_messages['invalid'], code='invalid')
        else:
            return range_value