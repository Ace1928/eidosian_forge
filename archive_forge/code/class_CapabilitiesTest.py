from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.capabilities import Capabilities
class CapabilitiesTest(utils.TestCase):

    def test_get_capabilities(self):
        capabilities = cs.capabilities.get('host')
        cs.assert_called('GET', '/capabilities/host')
        self.assertEqual(FAKE_CAPABILITY, capabilities._info)
        self._assert_request_id(capabilities)

    def test___repr__(self):
        """
        Unit test for Capabilities.__repr__

        Verify that Capabilities object can be printed.
        """
        cap = Capabilities(None, FAKE_CAPABILITY)
        self.assertEqual('<Capabilities: %s>' % FAKE_CAPABILITY['namespace'], repr(cap))

    def test__repr__when_empty(self):
        cap = Capabilities(None, {})
        self.assertEqual('<Capabilities: None>', repr(cap))