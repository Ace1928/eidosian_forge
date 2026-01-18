import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
class CinderVolumeTestsWithParameters(base.ClientTestBase):
    """Check of cinder volume create commands with parameters."""

    def test_volume_create_description(self):
        """Test steps:

        1) create volume with description
        2) check that volume has right description
        """
        volume_description = 'test_description'
        volume = self.object_create('volume', params='--description {0} 1'.format(volume_description))
        self.assertEqual(volume_description, volume['description'])

    def test_volume_create_metadata(self):
        """Test steps:

        1) create volume with metadata
        2) check that metadata complies entered
        """
        volume = self.object_create('volume', params='--metadata test_metadata=test_date 1')
        self.assertEqual(str({'test_metadata': 'test_date'}), volume['metadata'])