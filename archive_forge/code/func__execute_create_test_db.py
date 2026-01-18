import os
import subprocess
import sys
from django.db.backends.base.creation import BaseDatabaseCreation
from .client import DatabaseClient
def _execute_create_test_db(self, cursor, parameters, keepdb=False):
    try:
        super()._execute_create_test_db(cursor, parameters, keepdb)
    except Exception as e:
        if len(e.args) < 1 or e.args[0] != 1007:
            self.log('Got an error creating the test database: %s' % e)
            sys.exit(2)
        else:
            raise