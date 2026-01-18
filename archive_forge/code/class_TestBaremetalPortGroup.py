import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortGroup(baremetal_fakes.TestBaremetal):

    def setUp(self):
        super(TestBaremetalPortGroup, self).setUp()
        self.baremetal_mock = self.app.client_manager.baremetal
        self.baremetal_mock.reset_mock()