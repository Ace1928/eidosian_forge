import unittest
from traits.api import (
class CustomStrType(TraitType):
    default_value_type = DefaultValue.constant
    default_value = 'a string value'

    def validate(self, obj, name, value):
        if not isinstance(value, Str):
            return value
        self.error(obj, name, value)