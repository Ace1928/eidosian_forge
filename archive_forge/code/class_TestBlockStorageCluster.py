from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
class TestBlockStorageCluster(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.cluster_mock = self.volume_client.clusters
        self.cluster_mock.reset_mock()