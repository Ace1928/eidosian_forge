import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalVolumeConnector(baremetal_fakes.TestBaremetal):

    def setUp(self):
        super(TestBaremetalVolumeConnector, self).setUp()
        self.baremetal_mock = self.app.client_manager.baremetal
        self.baremetal_mock.reset_mock()