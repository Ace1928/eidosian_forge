from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from sqlalchemy.sql import sqltypes
from .base import AddColumn
from .base import alter_table
from .base import ColumnComment
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import DefaultImpl
from ..util.sqla_compat import compiles
class OracleImpl(DefaultImpl):
    __dialect__ = 'oracle'
    transactional_ddl = False
    batch_separator = '/'
    command_terminator = ''
    type_synonyms = DefaultImpl.type_synonyms + ({'VARCHAR', 'VARCHAR2'}, {'BIGINT', 'INTEGER', 'SMALLINT', 'DECIMAL', 'NUMERIC', 'NUMBER'}, {'DOUBLE', 'FLOAT', 'DOUBLE_PRECISION'})
    identity_attrs_ignore = ()

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.batch_separator = self.context_opts.get('oracle_batch_separator', self.batch_separator)

    def _exec(self, construct: Any, *args, **kw) -> Optional[CursorResult]:
        result = super()._exec(construct, *args, **kw)
        if self.as_sql and self.batch_separator:
            self.static_output(self.batch_separator)
        return result

    def compare_server_default(self, inspector_column, metadata_column, rendered_metadata_default, rendered_inspector_default):
        if rendered_metadata_default is not None:
            rendered_metadata_default = re.sub('^\\((.+)\\)$', '\\1', rendered_metadata_default)
            rendered_metadata_default = re.sub('^\\"?\'(.+)\'\\"?$', '\\1', rendered_metadata_default)
        if rendered_inspector_default is not None:
            rendered_inspector_default = re.sub('^\\((.+)\\)$', '\\1', rendered_inspector_default)
            rendered_inspector_default = re.sub('^\\"?\'(.+)\'\\"?$', '\\1', rendered_inspector_default)
            rendered_inspector_default = rendered_inspector_default.strip()
        return rendered_inspector_default != rendered_metadata_default

    def emit_begin(self) -> None:
        self._exec('SET TRANSACTION READ WRITE')

    def emit_commit(self) -> None:
        self._exec('COMMIT')