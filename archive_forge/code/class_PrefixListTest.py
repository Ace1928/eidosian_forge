import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class PrefixListTest(AnyTraitTest):

    def setUp(self):
        self.obj = PrefixListTrait()
    _default_value = 'one'
    _good_values = ['o', 'on', 'one', 'tw', 'two', 'th', 'thr', 'thre', 'three']
    _bad_values = ['t', 'one ', ' two', 1, None]

    def coerce(self, value):
        return {'o': 'one', 'on': 'one', 'tw': 'two', 'th': 'three'}[value[:2]]