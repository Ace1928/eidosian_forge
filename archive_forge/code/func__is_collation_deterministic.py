from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _is_collation_deterministic(self, collation_name):
    with self.connection.cursor() as cursor:
        cursor.execute('\n                SELECT collisdeterministic\n                FROM pg_collation\n                WHERE collname = %s\n                ', [collation_name])
        row = cursor.fetchone()
        return row[0] if row else None