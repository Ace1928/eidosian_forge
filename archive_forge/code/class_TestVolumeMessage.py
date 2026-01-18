from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
class TestVolumeMessage(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.projects_mock = self.app.client_manager.identity.projects
        self.projects_mock.reset_mock()
        self.volume_messages_mock = self.volume_client.messages
        self.volume_messages_mock.reset_mock()