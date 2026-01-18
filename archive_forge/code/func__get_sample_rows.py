from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union
import sqlalchemy
from langchain_core._api import deprecated
from langchain_core.utils import get_from_env
from sqlalchemy import (
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
def _get_sample_rows(self, table: Table) -> str:
    command = select(table).limit(self._sample_rows_in_table_info)
    columns_str = '\t'.join([col.name for col in table.columns])
    try:
        with self._engine.connect() as connection:
            sample_rows_result = connection.execute(command)
            sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result))
        sample_rows_str = '\n'.join(['\t'.join(row) for row in sample_rows])
    except ProgrammingError:
        sample_rows_str = ''
    return f'{self._sample_rows_in_table_info} rows from {table.name} table:\n{columns_str}\n{sample_rows_str}'