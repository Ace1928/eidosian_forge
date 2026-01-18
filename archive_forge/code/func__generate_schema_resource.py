import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def _generate_schema_resource(self, database_resource):
    return provision.SchemaResource(database_resource, None if not self._BUILD_SCHEMA else self.generate_schema_create_all if not self._BUILD_WITH_MIGRATIONS else self.generate_schema_migrations, self._DROP_SCHEMA_PER_TEST)