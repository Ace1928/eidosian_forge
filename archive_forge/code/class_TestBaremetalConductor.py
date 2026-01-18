import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalConductor(baremetal_fakes.TestBaremetal):

    def setUp(self):
        super(TestBaremetalConductor, self).setUp()
        self.baremetal_mock = self.app.client_manager.baremetal
        self.baremetal_mock.reset_mock()