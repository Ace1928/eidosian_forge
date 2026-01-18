import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class EnumCollectionGUIExample(EnumCollectionExample):
    int_set_enum = Enum('int', 'set')
    correct_int_set_enum = Enum('int', 'set')