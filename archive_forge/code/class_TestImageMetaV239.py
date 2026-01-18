from novaclient.tests.functional import base
class TestImageMetaV239(base.ClientTestBase):
    """Functional tests for image-meta proxy API."""
    COMPUTE_API_VERSION = '2.39'

    def test_limits(self):
        """Tests that 2.39 won't return 'maxImageMeta' resource limit and
        the CLI output won't show it.
        """
        output = self.nova('limits')
        self.assertRaises(ValueError, self._get_value_from_the_table, output, 'maxImageMeta')