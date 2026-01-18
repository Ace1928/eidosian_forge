import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _destroy_test_user(self, cursor, parameters, verbosity):
    if verbosity >= 2:
        self.log('_destroy_test_user(): user=%s' % parameters['user'])
        self.log('Be patient. This can take some time...')
    statements = ['DROP USER %(user)s CASCADE']
    self._execute_statements(cursor, statements, parameters, verbosity)