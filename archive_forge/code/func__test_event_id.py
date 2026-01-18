from heat.common import identifier
from heat.tests import common
def _test_event_id(self, event_id):
    ei = identifier.EventIdentifier('t', 's', 'i', '/resources/p', event_id)
    self.assertEqual(event_id, ei.event_id)