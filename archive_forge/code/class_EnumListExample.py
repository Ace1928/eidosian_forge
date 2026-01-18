import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class EnumListExample(HasTraits):
    values = List(['foo', 'bar', 'baz'])
    value = Enum(['foo', 'bar', 'baz'])
    value_default = Enum('bar', ['foo', 'bar', 'baz'])
    value_name = Enum(values='values')
    value_name_default = Enum('bar', values='values')