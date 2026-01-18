from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes
def _index_columns(self, table, columns, col_suffixes, opclasses):
    if opclasses:
        return IndexColumns(table, columns, self.quote_name, col_suffixes=col_suffixes, opclasses=opclasses)
    return super()._index_columns(table, columns, col_suffixes, opclasses)