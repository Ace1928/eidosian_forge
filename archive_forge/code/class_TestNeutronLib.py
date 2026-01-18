import copy
from neutron_lib import constants
from neutron_lib.tests import _base as base
class TestNeutronLib(base.BaseTestCase):

    def test_sentinel_constant(self):
        foo = constants.Sentinel()
        bar = copy.deepcopy(foo)
        self.assertEqual(id(foo), id(bar))

    def test_sentinel_copy(self):
        singleton = constants.Sentinel()
        self.assertEqual(copy.deepcopy(singleton), copy.copy(singleton))