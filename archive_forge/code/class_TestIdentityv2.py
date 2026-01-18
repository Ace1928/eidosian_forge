import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class TestIdentityv2(utils.TestCommand):

    def setUp(self):
        super(TestIdentityv2, self).setUp()
        self.app.client_manager.identity = FakeIdentityv2Client(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)