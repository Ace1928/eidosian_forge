from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class QuotaClassSetsTest2_57(QuotaClassSetsTest2_50):
    """Tests the quota classes API binding using the 2.57 microversion."""
    api_version = '2.57'

    def setUp(self):
        super(QuotaClassSetsTest2_57, self).setUp()
        self.invalid_resources.extend(['injected_files', 'injected_file_content_bytes', 'injected_file_path_bytes'])

    def test_update_quota_invalid_resources(self):
        """Tests trying to update quota class values for invalid resources.

        This will fail with TypeError because the file-related resource
        kwargs aren't defined.
        """
        q = super(QuotaClassSetsTest2_57, self).test_update_quota_invalid_resources()
        self.assertRaises(TypeError, q.update, injected_files=1)
        self.assertRaises(TypeError, q.update, injected_file_content_bytes=1)
        self.assertRaises(TypeError, q.update, injected_file_path_bytes=1)