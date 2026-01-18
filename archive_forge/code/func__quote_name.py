import sys
from django.core.exceptions import ImproperlyConfigured
from django.db.backends.base.creation import BaseDatabaseCreation
from django.db.backends.postgresql.psycopg_any import errors
from django.db.backends.utils import strip_quotes
def _quote_name(self, name):
    return self.connection.ops.quote_name(name)