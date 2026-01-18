from django.apps import apps
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.encoding import is_protected_type
def _value_from_field(self, obj, field):
    value = field.value_from_object(obj)
    return value if is_protected_type(value) else field.value_to_string(obj)