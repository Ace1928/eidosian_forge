import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class ImaginaryValueTest(AnyTraitTest):

    def setUp(self):
        self.obj = ImaginaryValueTrait()
    _default_value = 99.0 - 99j
    _good_values = [10, -10, 10.1, -10.1, '10', '-10', '10.1', '-10.1', 10j, 10 + 10j, 10 - 10j, 10.1j, 10.1 + 10.1j, 10.1 - 10.1j, '10j', '10+10j', '10-10j']
    _bad_values = [b'10L', '-10L', 'ten', [10], {'ten': 10}, (10,), None]

    def coerce(self, value):
        return complex(value)