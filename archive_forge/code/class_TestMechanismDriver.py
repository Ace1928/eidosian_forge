from neutron_lib.plugins.ml2 import api
from neutron_lib.tests import _base as base
class TestMechanismDriver(base.BaseTestCase):

    def test__supports_port_binding(self):
        self.assertTrue(_MechanismDriver()._supports_port_binding)

    def test_get_workers(self):
        self.assertEqual((), _MechanismDriver().get_workers())

    def test_filter_hosts_with_segment_access(self):
        dummy_token = ['X']
        self.assertEqual(dummy_token, _MechanismDriver().filter_hosts_with_segment_access(dummy_token, dummy_token, dummy_token, dummy_token))