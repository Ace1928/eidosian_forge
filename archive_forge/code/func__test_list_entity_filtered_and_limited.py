import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _test_list_entity_filtered_and_limited(self, entity):
    self.config_fixture.config(list_limit=10)
    hints = driver_hints.Hints()
    entities = self._list_entities(entity)(hints=hints)
    self.assertEqual(hints.limit['limit'], len(entities))
    self.assertTrue(hints.limit['truncated'])
    if entity == 'project':
        self.config_fixture.config(group='resource', list_limit=5)
    else:
        self.config_fixture.config(group='identity', list_limit=5)
    hints = driver_hints.Hints()
    entities = self._list_entities(entity)(hints=hints)
    self.assertEqual(hints.limit['limit'], len(entities))
    entities = self._list_entities(entity)()
    self.assertGreaterEqual(len(entities), 20)
    self._match_with_list(self.entity_lists[entity], entities)