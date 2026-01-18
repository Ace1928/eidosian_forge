import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
class TestNormalization(TestCase):
    """Verify that we have our normalizations correct."""

    def test_normalize(self):
        self.assertEqual(a_circle_d, normalize('NFD', a_circle_c))
        self.assertEqual(a_circle_c, normalize('NFC', a_circle_d))
        self.assertEqual(a_dots_d, normalize('NFD', a_dots_c))
        self.assertEqual(a_dots_c, normalize('NFC', a_dots_d))
        self.assertEqual(z_umlat_d, normalize('NFD', z_umlat_c))
        self.assertEqual(z_umlat_c, normalize('NFC', z_umlat_d))
        self.assertEqual(squared_d, normalize('NFC', squared_c))
        self.assertEqual(squared_c, normalize('NFD', squared_d))
        self.assertEqual(quarter_d, normalize('NFC', quarter_c))
        self.assertEqual(quarter_c, normalize('NFD', quarter_d))