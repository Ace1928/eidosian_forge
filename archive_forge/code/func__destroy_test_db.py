import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def _destroy_test_db(self, test_database_name, verbosity):
    """
        Internal implementation - remove the test db tables.
        """
    with self._nodb_cursor() as cursor:
        cursor.execute('DROP DATABASE %s' % self.connection.ops.quote_name(test_database_name))