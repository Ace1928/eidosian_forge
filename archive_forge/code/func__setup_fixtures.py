import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def _setup_fixtures(self):
    testresources.setUpResources(self, self.resources, testresources._get_result())
    self.addCleanup(testresources.tearDownResources, self, self.resources, testresources._get_result())
    fixtures = self._instantiate_fixtures()
    for fixture in fixtures:
        self.useFixture(fixture)
        if not fixture._has_db_resource():
            msg = fixture._get_db_resource_not_available_reason()
            if self.SKIP_ON_UNAVAILABLE_DB:
                self.skipTest(msg)
            else:
                self.fail(msg)