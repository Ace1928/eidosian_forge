import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def _create_test_db(self, verbosity, autoclobber, keepdb=False):
    """
        Internal implementation - create the test db tables.
        """
    test_database_name = self._get_test_db_name()
    test_db_params = {'dbname': self.connection.ops.quote_name(test_database_name), 'suffix': self.sql_table_creation_suffix()}
    with self._nodb_cursor() as cursor:
        try:
            self._execute_create_test_db(cursor, test_db_params, keepdb)
        except Exception as e:
            if keepdb:
                return test_database_name
            self.log('Got an error creating the test database: %s' % e)
            if not autoclobber:
                confirm = input("Type 'yes' if you would like to try deleting the test database '%s', or 'no' to cancel: " % test_database_name)
            if autoclobber or confirm == 'yes':
                try:
                    if verbosity >= 1:
                        self.log('Destroying old test database for alias %s...' % (self._get_database_display_str(verbosity, test_database_name),))
                    cursor.execute('DROP DATABASE %(dbname)s' % test_db_params)
                    self._execute_create_test_db(cursor, test_db_params, keepdb)
                except Exception as e:
                    self.log('Got an error recreating the test database: %s' % e)
                    sys.exit(2)
            else:
                self.log('Tests cancelled.')
                sys.exit(1)
    return test_database_name