import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
@requires_traitsui
class TestDeprecatedTraits(unittest.TestCase):

    def test_color_deprecated(self):
        with self.assertWarnsRegex(DeprecationWarning, "'Color' in 'traits'"):
            Color()

    def test_rgb_color_deprecated(self):
        with self.assertWarnsRegex(DeprecationWarning, "'RGBColor' in 'traits'"):
            RGBColor()

    def test_font_deprecated(self):
        with self.assertWarnsRegex(DeprecationWarning, "'Font' in 'traits'"):
            Font()