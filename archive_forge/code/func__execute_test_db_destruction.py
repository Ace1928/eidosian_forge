import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _execute_test_db_destruction(self, cursor, parameters, verbosity):
    if verbosity >= 2:
        self.log('_execute_test_db_destruction(): dbname=%s' % parameters['user'])
    statements = ['DROP TABLESPACE %(tblspace)s INCLUDING CONTENTS AND DATAFILES CASCADE CONSTRAINTS', 'DROP TABLESPACE %(tblspace_temp)s INCLUDING CONTENTS AND DATAFILES CASCADE CONSTRAINTS']
    self._execute_statements(cursor, statements, parameters, verbosity)