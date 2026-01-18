import gyp.common
import unittest
import sys
class TestGetFlavor(unittest.TestCase):
    """Test that gyp.common.GetFlavor works as intended"""
    original_platform = ''

    def setUp(self):
        self.original_platform = sys.platform

    def tearDown(self):
        sys.platform = self.original_platform

    def assertFlavor(self, expected, argument, param):
        sys.platform = argument
        self.assertEqual(expected, gyp.common.GetFlavor(param))

    def test_platform_default(self):
        self.assertFlavor('freebsd', 'freebsd9', {})
        self.assertFlavor('freebsd', 'freebsd10', {})
        self.assertFlavor('openbsd', 'openbsd5', {})
        self.assertFlavor('solaris', 'sunos5', {})
        self.assertFlavor('solaris', 'sunos', {})
        self.assertFlavor('linux', 'linux2', {})
        self.assertFlavor('linux', 'linux3', {})
        self.assertFlavor('linux', 'linux', {})

    def test_param(self):
        self.assertFlavor('foobar', 'linux2', {'flavor': 'foobar'})