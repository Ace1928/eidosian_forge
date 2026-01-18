from ... import bedding, errors, osutils
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ModuleAvailableFeature
class TestDependencyManagement(TestCase):
    """Tests for managing the dependency on launchpadlib."""
    _test_needs_features = [launchpadlib_feature]

    def setUp(self):
        super().setUp()
        from . import lp_api
        self.lp_api = lp_api

    def patch(self, obj, name, value):
        """Temporarily set the 'name' attribute of 'obj' to 'value'."""
        self.overrideAttr(obj, name, value)

    def test_get_launchpadlib_version(self):
        version_info = self.lp_api.parse_launchpadlib_version('1.5.1')
        self.assertEqual((1, 5, 1), version_info)

    def test_supported_launchpadlib_version(self):
        launchpadlib = launchpadlib_feature.module
        self.patch(launchpadlib, '__version__', '1.5.1')
        self.lp_api.MINIMUM_LAUNCHPADLIB_VERSION = (1, 5, 1)
        self.lp_api.check_launchpadlib_compatibility()

    def test_unsupported_launchpadlib_version(self):
        launchpadlib = launchpadlib_feature.module
        self.patch(launchpadlib, '__version__', '1.5.0')
        self.lp_api.MINIMUM_LAUNCHPADLIB_VERSION = (1, 5, 1)
        self.assertRaises(errors.DependencyNotPresent, self.lp_api.check_launchpadlib_compatibility)