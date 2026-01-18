from heat.common import identifier
from heat.tests import common
def _test_event_init(self, event_id):
    si = identifier.HeatIdentifier('t', 's', 'i')
    pi = identifier.ResourceIdentifier(resource_name='p', **si)
    ei = identifier.EventIdentifier(event_id=event_id, **pi)
    self.assertEqual('/resources/p/events/{0}'.format(event_id), ei.path)