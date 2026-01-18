import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _test_database_passwd(self):
    password = self._test_settings_get('PASSWORD')
    if password is None and self._test_user_create():
        password = get_random_string(30)
    return password