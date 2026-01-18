from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _alter_column_collation_sql(self, model, new_field, new_type, new_collation, old_field):
    sql = self.sql_alter_column_collate
    if (using_sql := self._using_sql(new_field, old_field)):
        sql += using_sql
    return (sql % {'column': self.quote_name(new_field.column), 'type': new_type, 'collation': ' ' + self._collate_sql(new_collation) if new_collation else ''}, [])