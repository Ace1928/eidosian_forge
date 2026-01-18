from ... import bedding, errors, osutils
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ModuleAvailableFeature
class TestCacheDirectory(TestCase):
    """Tests for get_cache_directory."""
    _test_needs_features = [launchpadlib_feature]

    def test_get_cache_directory(self):
        from . import lp_api
        try:
            expected_path = osutils.pathjoin(bedding.cache_dir(), 'launchpad')
        except OSError:
            self.assertRaises(EnvironmentError, lp_api.get_cache_directory)
        else:
            self.assertEqual(expected_path, lp_api.get_cache_directory())