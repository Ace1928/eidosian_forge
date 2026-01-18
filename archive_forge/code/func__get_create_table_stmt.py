from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, List, Optional
def _get_create_table_stmt(self, table: str) -> str:
    statement = self._spark.sql(f'SHOW CREATE TABLE {table}').collect()[0].createtab_stmt
    using_clause_index = statement.find('USING')
    return statement[:using_clause_index] + ';'