from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _get_sequence_name(self, table, column):
    with self.connection.cursor() as cursor:
        for sequence in self.connection.introspection.get_sequences(cursor, table):
            if sequence['column'] == column:
                return sequence['name']
    return None