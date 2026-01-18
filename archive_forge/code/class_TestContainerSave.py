import copy
from requests_mock.contrib import fixture
from openstackclient.object.v1 import container as container_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
class TestContainerSave(TestContainerAll):

    def setUp(self):
        super(TestContainerSave, self).setUp()
        self.cmd = container_cmds.SaveContainer(self.app, None)