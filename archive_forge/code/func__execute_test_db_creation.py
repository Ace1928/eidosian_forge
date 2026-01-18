import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _execute_test_db_creation(self, cursor, parameters, verbosity, keepdb=False):
    if verbosity >= 2:
        self.log('_create_test_db(): dbname = %s' % parameters['user'])
    if self._test_database_oracle_managed_files():
        statements = ['\n                CREATE TABLESPACE %(tblspace)s\n                DATAFILE SIZE %(size)s\n                AUTOEXTEND ON NEXT %(extsize)s MAXSIZE %(maxsize)s\n                ', '\n                CREATE TEMPORARY TABLESPACE %(tblspace_temp)s\n                TEMPFILE SIZE %(size_tmp)s\n                AUTOEXTEND ON NEXT %(extsize_tmp)s MAXSIZE %(maxsize_tmp)s\n                ']
    else:
        statements = ["\n                CREATE TABLESPACE %(tblspace)s\n                DATAFILE '%(datafile)s' SIZE %(size)s REUSE\n                AUTOEXTEND ON NEXT %(extsize)s MAXSIZE %(maxsize)s\n                ", "\n                CREATE TEMPORARY TABLESPACE %(tblspace_temp)s\n                TEMPFILE '%(datafile_tmp)s' SIZE %(size_tmp)s REUSE\n                AUTOEXTEND ON NEXT %(extsize_tmp)s MAXSIZE %(maxsize_tmp)s\n                "]
    acceptable_ora_err = 'ORA-01543' if keepdb else None
    self._execute_allow_fail_statements(cursor, statements, parameters, verbosity, acceptable_ora_err)