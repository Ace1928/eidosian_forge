from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad import Schema, assert_or_throw
from fugue import DataFrame, IterableDataFrame, LocalBoundedDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import drop_columns, get_column_names, is_df, rename
from ._compat import IbisSchema, IbisTable
from ._utils import pa_to_ibis_type, to_schema
def _alter_table_columns(self, table: IbisTable, new_schema: Schema) -> IbisTable:
    fields: Dict[str, Any] = {}
    schema = table.schema()
    for _name, _type, f2 in zip(schema.names, schema.types, new_schema.fields):
        _new_name, _new_type = (f2.name, pa_to_ibis_type(f2.type))
        assert_or_throw(_name == _new_name, lambda: ValueError(f'schema name mismatch: {_name} vs {_new_name}'))
        if _type == _new_type:
            continue
        else:
            fields[_name] = table[_name].cast(_new_type)
    if len(fields) == 0:
        return table
    return table.mutate(**fields)