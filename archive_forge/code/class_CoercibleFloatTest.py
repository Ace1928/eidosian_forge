import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class CoercibleFloatTest(AnyTraitTest):

    def setUp(self):
        self.obj = CoercibleFloatTrait()
    _default_value = 99.0
    _good_values = [10, -10, 10.1, -10.1, '10', '-10', '10.1', '-10.1', b'10', b'-10', b'10.1', b'-10.1']
    _bad_values = ['10L', '-10L', b'10L', b'-10L', 'ten', b'ten', [10], {'ten': 10}, (10,), None, 1j]

    def coerce(self, value):
        return float(value)