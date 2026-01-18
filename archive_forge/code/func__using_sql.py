from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _using_sql(self, new_field, old_field):
    using_sql = ' USING %(column)s::%(type)s'
    new_internal_type = new_field.get_internal_type()
    old_internal_type = old_field.get_internal_type()
    if new_internal_type == 'ArrayField' and new_internal_type == old_internal_type:
        if list(self._field_base_data_types(old_field)) != list(self._field_base_data_types(new_field)):
            return using_sql
    elif self._field_data_type(old_field) != self._field_data_type(new_field):
        return using_sql
    return ''