import webob.exc
from neutron_lib.api import faults
from neutron_lib.tests import _base as base
class TestFaultMap(base.BaseTestCase):

    def test_extend_fault_map(self):
        fault_map_dict = {NotImplemented: webob.exc.HTTPServiceUnavailable}
        faults.FAULT_MAP.update(fault_map_dict)
        self.assertIn(NotImplemented, faults.FAULT_MAP)