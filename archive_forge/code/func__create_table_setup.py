from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def _create_table_setup(self):
    """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
    column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
    escape = _get_valid_sqlite_name
    create_tbl_stmts = [escape(cname) + ' ' + ctype for cname, ctype, _ in column_names_and_types]
    if self.keys is not None and len(self.keys):
        if not is_list_like(self.keys):
            keys = [self.keys]
        else:
            keys = self.keys
        cnames_br = ', '.join([escape(c) for c in keys])
        create_tbl_stmts.append(f'CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})')
    if self.schema:
        schema_name = self.schema + '.'
    else:
        schema_name = ''
    create_stmts = ['CREATE TABLE ' + schema_name + escape(self.name) + ' (\n' + ',\n  '.join(create_tbl_stmts) + '\n)']
    ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
    if len(ix_cols):
        cnames = '_'.join(ix_cols)
        cnames_br = ','.join([escape(c) for c in ix_cols])
        create_stmts.append('CREATE INDEX ' + escape('ix_' + self.name + '_' + cnames) + 'ON ' + escape(self.name) + ' (' + cnames_br + ')')
    return create_stmts