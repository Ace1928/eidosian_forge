from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _delete_index_sql(self, model, name, sql=None, concurrently=False):
    sql = self.sql_delete_index_concurrently if concurrently else self.sql_delete_index
    return super()._delete_index_sql(model, name, sql)