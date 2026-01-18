import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class ResetsData(object):
    """Mixin defining a fixture that resets schema data without dropping."""
    _DROP_SCHEMA_PER_TEST = False

    def setup_for_reset(self, engine, enginefacade):
        """"Perform setup that may be needed before the test runs."""

    def reset_schema_data(self, engine, enginefacade):
        """Reset the data in the schema."""