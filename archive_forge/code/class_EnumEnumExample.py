import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class EnumEnumExample(HasTraits):
    values = Any(FooEnum)
    value = Enum(FooEnum)
    value_default = Enum(FooEnum.bar, FooEnum)
    value_name = Enum(values='values')
    value_name_default = Enum(FooEnum.bar, values='values')