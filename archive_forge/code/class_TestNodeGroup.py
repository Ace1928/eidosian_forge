import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestNodeGroup(magnum_fakes.TestMagnumClientOSCV1):

    def setUp(self):
        super(TestNodeGroup, self).setUp()
        self.ng_mock = self.app.client_manager.container_infra.nodegroups