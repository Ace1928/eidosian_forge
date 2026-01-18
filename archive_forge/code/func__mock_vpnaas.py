import testtools
from osc_lib import exceptions
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def _mock_vpnaas(*args, **kwargs):
    return {'id': args[0]}