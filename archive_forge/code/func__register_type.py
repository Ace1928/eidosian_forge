from functools import lru_cache
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.base import DatabaseWrapper as PsycopgDatabaseWrapper
from django.db.backends.postgresql.features import (
from django.db.backends.postgresql.introspection import (
from django.db.backends.postgresql.operations import (
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from .adapter import PostGISAdapter
from .features import DatabaseFeatures
from .introspection import PostGISIntrospection
from .operations import PostGISOperations
from .schema import PostGISSchemaEditor
def _register_type(self, pg_connection, typename):
    registry = self._type_infos[typename]
    try:
        info = registry[self.alias]
    except KeyError:
        info = TypeInfo.fetch(pg_connection, typename)
        registry[self.alias] = info
    if info:
        info.register(pg_connection)
        pg_connection.adapters.register_loader(info.oid, TextLoader)
        pg_connection.adapters.register_loader(info.oid, TextBinaryLoader)
    return info.oid if info else None