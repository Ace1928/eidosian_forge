from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _field_data_type(self, field):
    if field.is_relation:
        return field.rel_db_type(self.connection)
    return self.connection.data_types.get(field.get_internal_type(), field.db_type(self.connection))