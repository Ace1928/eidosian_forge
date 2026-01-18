from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
class TestConsistencyGroupSnapshot(volume_fakes.TestVolume):

    def setUp(self):
        super(TestConsistencyGroupSnapshot, self).setUp()
        self.cgsnapshots_mock = self.volume_client.cgsnapshots
        self.cgsnapshots_mock.reset_mock()
        self.consistencygroups_mock = self.volume_client.consistencygroups
        self.consistencygroups_mock.reset_mock()