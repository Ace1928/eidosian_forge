from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, List, Optional
def _get_all_table_names(self) -> Iterable[str]:
    rows = self._spark.sql('SHOW TABLES').select('tableName').collect()
    return list(map(lambda row: row.tableName, rows))