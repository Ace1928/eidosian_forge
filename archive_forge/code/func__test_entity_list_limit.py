import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def _test_entity_list_limit(self, entity, driver):
    """GET /<entities> (limited).

        Test Plan:

        - For the specified type of entity:
            - Update policy for no protection on api
            - Add a bunch of entities
            - Set the global list limit to 5, and check that getting all
            - entities only returns 5
            - Set the driver list_limit to 4, and check that now only 4 are
            - returned

        """
    if entity == 'policy':
        plural = 'policies'
    else:
        plural = '%ss' % entity
    self._set_policy({'identity:list_%s' % plural: []})
    self.config_fixture.config(list_limit=5)
    self.config_fixture.config(group=driver, list_limit=None)
    r = self.get('/%s' % plural, auth=self.auth)
    self.assertEqual(5, len(r.result.get(plural)))
    self.assertIs(r.result.get('truncated'), True)
    self.config_fixture.config(group=driver, list_limit=4)
    r = self.get('/%s' % plural, auth=self.auth)
    self.assertEqual(4, len(r.result.get(plural)))
    self.assertIs(r.result.get('truncated'), True)