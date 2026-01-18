import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class RequireIfMatching(FormValidator):
    """
    Require a list of fields based on the value of another field.

    This validator is applied to a form, not an individual field (usually
    using a Schema's ``pre_validators`` or ``chained_validators``).

    You provide a field name, an expected value and a list of required fields
    (a list of string key names). If the value of the field, if present,
    matches the value of ``expected_value``, then the validator will raise an
    ``Invalid`` exception for every field in ``required_fields`` that is
    missing.

    ::

        >>> from formencode import validators
        >>> v = validators.RequireIfMatching('phone_type', expected_value='mobile', required_fields=['mobile'])
        >>> v.to_python(dict(phone_type='mobile'))
        Traceback (most recent call last):
            ...
        Invalid: You must give a value for mobile
        >>> v.to_python(dict(phone_type='someothervalue'))
        {'phone_type': 'someothervalue'}
    """
    field = None
    expected_value = None
    required_fields = []
    __unpackargs__ = ('field', 'expected_value')

    def _convert_to_python(self, value_dict, state):
        is_empty = self.field_is_empty
        if self.field in value_dict and value_dict.get(self.field) == self.expected_value:
            for required_field in self.required_fields:
                if required_field not in value_dict or is_empty(value_dict.get(required_field)):
                    raise Invalid(_('You must give a value for %s') % required_field, value_dict, state, error_dict={required_field: Invalid(self.message('empty', state), value_dict.get(required_field), state)})
        return value_dict