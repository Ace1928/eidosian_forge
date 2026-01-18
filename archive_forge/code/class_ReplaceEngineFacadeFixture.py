import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class ReplaceEngineFacadeFixture(fixtures.Fixture):
    """A fixture that will plug the engine of one enginefacade into another.

    This fixture can be used by test suites that already have their own non-
    oslo_db database setup / teardown schemes, to plug any URL or test-oriented
    enginefacade as-is into an enginefacade-oriented API.

    For applications that use oslo.db's testing fixtures, the
    ReplaceEngineFacade fixture is used internally.

    E.g.::

        class MyDBTest(TestCase):

            def setUp(self):
                from myapplication.api import main_enginefacade

                my_test_enginefacade = enginefacade.transaction_context()
                my_test_enginefacade.configure(connection=my_test_url)

                self.useFixture(
                    ReplaceEngineFacadeFixture(
                        main_enginefacade, my_test_enginefacade))

    Above, the main_enginefacade object is the normal application level
    one, and my_test_enginefacade is a local one that we've created to
    refer to some testing database.   Throughout the fixture's setup,
    the application level enginefacade will use the engine factory and
    engines of the testing enginefacade, and at fixture teardown will be
    replaced back.

    """

    def __init__(self, enginefacade, replace_with_enginefacade):
        super(ReplaceEngineFacadeFixture, self).__init__()
        self.enginefacade = enginefacade
        self.replace_with_enginefacade = replace_with_enginefacade

    def _setUp(self):
        _reset_facade = self.enginefacade.patch_factory(self.replace_with_enginefacade._factory)
        self.addCleanup(_reset_facade)