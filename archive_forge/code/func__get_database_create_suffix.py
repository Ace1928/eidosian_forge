import sys
from django.core.exceptions import ImproperlyConfigured
from django.db.backends.base.creation import BaseDatabaseCreation
from django.db.backends.postgresql.psycopg_any import errors
from django.db.backends.utils import strip_quotes
def _get_database_create_suffix(self, encoding=None, template=None):
    suffix = ''
    if encoding:
        suffix += " ENCODING '{}'".format(encoding)
    if template:
        suffix += ' TEMPLATE {}'.format(self._quote_name(template))
    return suffix and 'WITH' + suffix