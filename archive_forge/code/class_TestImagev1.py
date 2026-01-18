from unittest import mock
import uuid
from openstack.image.v1 import _proxy
from openstack.image.v1 import image
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
class TestImagev1(FakeClientMixin, utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.volume = volume_fakes.FakeVolumev1Client(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)
        self.volume_client = self.app.client_manager.volume