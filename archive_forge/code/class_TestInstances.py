from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestInstances(fakes.TestDatabasev1):

    def setUp(self):
        super(TestInstances, self).setUp()
        self.fake_instances = fakes.FakeInstances()
        self.instance_client = self.app.client_manager.database.instances
        self.mgmt_client = self.app.client_manager.database.mgmt_instances