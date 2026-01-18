from heat.common import identifier
from heat.tests import common
class EventIdentifierTest(common.HeatTestCase):

    def test_event_init_integer_id(self):
        self._test_event_init('42')

    def test_event_init_uuid_id(self):
        self._test_event_init('a3455d8c-9f88-404d-a85b-5315293e67de')

    def _test_event_init(self, event_id):
        si = identifier.HeatIdentifier('t', 's', 'i')
        pi = identifier.ResourceIdentifier(resource_name='p', **si)
        ei = identifier.EventIdentifier(event_id=event_id, **pi)
        self.assertEqual('/resources/p/events/{0}'.format(event_id), ei.path)

    def test_event_init_from_dict(self):
        hi = identifier.HeatIdentifier('t', 's', 'i', '/resources/p/events/42')
        ei = identifier.EventIdentifier(**hi)
        self.assertEqual(hi, ei)

    def test_event_stack(self):
        si = identifier.HeatIdentifier('t', 's', 'i')
        pi = identifier.ResourceIdentifier(resource_name='r', **si)
        ei = identifier.EventIdentifier(event_id='e', **pi)
        self.assertEqual(si, ei.stack())

    def test_event_resource(self):
        si = identifier.HeatIdentifier('t', 's', 'i')
        pi = identifier.ResourceIdentifier(resource_name='r', **si)
        ei = identifier.EventIdentifier(event_id='e', **pi)
        self.assertEqual(pi, ei.resource())

    def test_resource_name(self):
        ei = identifier.EventIdentifier('t', 's', 'i', '/resources/p', 'e')
        self.assertEqual('p', ei.resource_name)

    def test_event_id_integer(self):
        self._test_event_id('42')

    def test_event_id_uuid(self):
        self._test_event_id('a3455d8c-9f88-404d-a85b-5315293e67de')

    def _test_event_id(self, event_id):
        ei = identifier.EventIdentifier('t', 's', 'i', '/resources/p', event_id)
        self.assertEqual(event_id, ei.event_id)