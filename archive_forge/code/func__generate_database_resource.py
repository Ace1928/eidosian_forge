import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def _generate_database_resource(self, _enginefacade):
    return provision.DatabaseResource(self.driver, _enginefacade, provision_new_database=True)