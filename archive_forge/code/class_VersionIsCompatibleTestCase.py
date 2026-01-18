from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class VersionIsCompatibleTestCase(test_utils.BaseTestCase):

    def test_version_is_compatible_same(self):
        self.assertTrue(utils.version_is_compatible('1.23', '1.23'))

    def test_version_is_compatible_newer_minor(self):
        self.assertTrue(utils.version_is_compatible('1.24', '1.23'))

    def test_version_is_compatible_older_minor(self):
        self.assertFalse(utils.version_is_compatible('1.22', '1.23'))

    def test_version_is_compatible_major_difference1(self):
        self.assertFalse(utils.version_is_compatible('2.23', '1.23'))

    def test_version_is_compatible_major_difference2(self):
        self.assertFalse(utils.version_is_compatible('1.23', '2.23'))

    def test_version_is_compatible_newer_rev(self):
        self.assertFalse(utils.version_is_compatible('1.23', '1.23.1'))

    def test_version_is_compatible_newer_rev_both(self):
        self.assertFalse(utils.version_is_compatible('1.23.1', '1.23.2'))

    def test_version_is_compatible_older_rev_both(self):
        self.assertTrue(utils.version_is_compatible('1.23.2', '1.23.1'))

    def test_version_is_compatible_older_rev(self):
        self.assertTrue(utils.version_is_compatible('1.24', '1.23.1'))

    def test_version_is_compatible_no_rev_is_zero(self):
        self.assertTrue(utils.version_is_compatible('1.23.0', '1.23'))