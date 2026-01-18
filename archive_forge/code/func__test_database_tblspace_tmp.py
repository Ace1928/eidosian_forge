import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _test_database_tblspace_tmp(self):
    settings_dict = self.connection.settings_dict
    return settings_dict['TEST'].get('TBLSPACE_TMP', TEST_DATABASE_PREFIX + settings_dict['USER'] + '_temp')