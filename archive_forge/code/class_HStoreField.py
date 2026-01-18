import json
from django.contrib.postgres import forms, lookups
from django.contrib.postgres.fields.array import ArrayField
from django.core import exceptions
from django.db.models import Field, TextField, Transform
from django.db.models.fields.mixins import CheckFieldDefaultMixin
from django.utils.translation import gettext_lazy as _
class HStoreField(CheckFieldDefaultMixin, Field):
    empty_strings_allowed = False
    description = _('Map of strings to strings/nulls')
    default_error_messages = {'not_a_string': _('The value of “%(key)s” is not a string or null.')}
    _default_hint = ('dict', '{}')

    def db_type(self, connection):
        return 'hstore'

    def get_transform(self, name):
        transform = super().get_transform(name)
        if transform:
            return transform
        return KeyTransformFactory(name)

    def validate(self, value, model_instance):
        super().validate(value, model_instance)
        for key, val in value.items():
            if not isinstance(val, str) and val is not None:
                raise exceptions.ValidationError(self.error_messages['not_a_string'], code='not_a_string', params={'key': key})

    def to_python(self, value):
        if isinstance(value, str):
            value = json.loads(value)
        return value

    def value_to_string(self, obj):
        return json.dumps(self.value_from_object(obj))

    def formfield(self, **kwargs):
        return super().formfield(**{'form_class': forms.HStoreField, **kwargs})

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if isinstance(value, dict):
            prep_value = {}
            for key, val in value.items():
                key = str(key)
                if val is not None:
                    val = str(val)
                prep_value[key] = val
            value = prep_value
        if isinstance(value, list):
            value = [str(item) for item in value]
        return value