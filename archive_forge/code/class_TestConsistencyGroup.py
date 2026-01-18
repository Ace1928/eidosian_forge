from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
class TestConsistencyGroup(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.consistencygroups_mock = self.volume_client.consistencygroups
        self.consistencygroups_mock.reset_mock()
        self.cgsnapshots_mock = self.volume_client.cgsnapshots
        self.cgsnapshots_mock.reset_mock()
        self.volumes_mock = self.volume_client.volumes
        self.volumes_mock.reset_mock()
        self.types_mock = self.volume_client.volume_types
        self.types_mock.reset_mock()