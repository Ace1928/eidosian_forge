import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def _check_make_delta(string1, string2):
    self.assertRaises(TypeError, self.make_delta, string1, string2)