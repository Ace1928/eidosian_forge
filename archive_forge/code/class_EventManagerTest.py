import testtools
from ironicclient.tests.unit import utils
from ironicclient.v1 import events
class EventManagerTest(testtools.TestCase):

    def setUp(self):
        super(EventManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = events.EventManager(self.api)

    def test_event(self):
        evts = self.mgr.create(**FAKE_EVENTS)
        expect = [('POST', '/v1/events', {}, FAKE_EVENTS)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(evts)

    def test_network_port_event(self):
        evts = self.mgr.create(**FAKE_NETWORK_PORT_EVENTS)
        expect = [('POST', '/v1/events', {}, FAKE_NETWORK_PORT_EVENTS)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(evts)