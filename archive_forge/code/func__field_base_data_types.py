from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _field_base_data_types(self, field):
    if field.base_field.get_internal_type() == 'ArrayField':
        yield from self._field_base_data_types(field.base_field)
    else:
        yield self._field_data_type(field.base_field)