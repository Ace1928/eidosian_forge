import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def _instantiate_fixtures(self):
    if self._instantiated_fixtures:
        return self._instantiated_fixtures
    self._instantiated_fixtures = utils.to_list(self.generate_fixtures())
    return self._instantiated_fixtures