import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
class TestMakeAndApplyCompatible(tests.TestCase):
    scenarios = two_way_scenarios()
    make_delta = None
    apply_delta = None

    def assertMakeAndApply(self, source, target):
        """Assert that generating a delta and applying gives success."""
        delta = self.make_delta(source, target)
        bytes = self.apply_delta(source, delta)
        self.assertEqualDiff(target, bytes)

    def test_direct(self):
        self.assertMakeAndApply(_text1, _text2)
        self.assertMakeAndApply(_text2, _text1)
        self.assertMakeAndApply(_text1, _text3)
        self.assertMakeAndApply(_text3, _text1)
        self.assertMakeAndApply(_text2, _text3)
        self.assertMakeAndApply(_text3, _text2)