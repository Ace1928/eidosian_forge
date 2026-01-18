import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class SimpleDbFixture(BaseDbFixture):
    """Fixture which provides an engine from a fixed URL.

    The SimpleDbFixture is generally appropriate only for a SQLite memory
    database, as this database is naturally isolated from other processes and
    does not require management of schemas.   For tests that need to
    run specifically against MySQL or Postgresql, the OpportunisticDbFixture
    is more appropriate.

    The database connection information itself comes from the provisoning
    system, matching the desired driver (typically sqlite) to the default URL
    that provisioning provides for this driver (in the case of sqlite, it's
    the SQLite memory URL, e.g. sqlite://.  For MySQL and Postgresql, it's
    the familiar "openstack_citest" URL on localhost).

    There are a variety of create/drop schemes that can take place:

    * The default is to procure a database connection on setup,
      and at teardown, an instruction is issued to "drop" all
      objects in the schema (e.g. tables, indexes).  The SQLAlchemy
      engine itself remains referenced at the class level for subsequent
      re-use.

    * When the GeneratesSchema or GeneratesSchemaFromMigrations mixins
      are implemented, the appropriate generate_schema method is also
      called when the fixture is set up, by default this is per test.

    * When the DeletesFromSchema mixin is implemented, the generate_schema
      method is now only called **once**, and the "drop all objects"
      system is replaced with the delete_from_schema method.   This
      allows the same database to remain set up with all schema objects
      intact, so that expensive migrations need not be run on every test.

    * The fixture does **not** dispose the engine at the end of a test.
      It is assumed the same engine will be re-used many times across
      many tests.  The AdHocDbFixture extends this one to provide
      engine.dispose() at the end of a test.

    This fixture is intended to work without needing a reference to
    the test itself, and therefore cannot take advantage of the
    OptimisingTestSuite.

    """
    _dependency_resources = {}

    def _get_provisioned_db(self):
        return self._dependency_resources['_db_%s' % self.ident]

    def _generate_database_resource(self, _enginefacade):
        return provision.DatabaseResource(self.driver, _enginefacade, provision_new_database=False)

    def _setUp(self):
        super(SimpleDbFixture, self)._setUp()
        cls = self.__class__
        if '_db_%s' % self.ident not in cls._dependency_resources:
            resources = self._get_resources()
            for name, resource in resources:
                cls._dependency_resources[name] = resource.getResource()
        provisioned_db = self._get_provisioned_db()
        if not self._DROP_SCHEMA_PER_TEST:
            self.setup_for_reset(provisioned_db.engine, provisioned_db.enginefacade)
        self.useFixture(ReplaceEngineFacadeFixture(self.get_per_test_enginefacade(), provisioned_db.enginefacade))
        if not self._DROP_SCHEMA_PER_TEST:
            self.addCleanup(self.reset_schema_data, provisioned_db.engine, provisioned_db.enginefacade)
        self.addCleanup(self._cleanup)

    def _teardown_resources(self):
        for name, resource in self._get_resources():
            dep = self._dependency_resources.pop(name)
            resource.finishedWith(dep)

    def _cleanup(self):
        pass