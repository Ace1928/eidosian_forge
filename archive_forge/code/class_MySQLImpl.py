from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema
from sqlalchemy import types as sqltypes
from .base import alter_table
from .base import AlterColumn
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .impl import DefaultImpl
from .. import util
from ..util import sqla_compat
from ..util.sqla_compat import _is_mariadb
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import compiles
class MySQLImpl(DefaultImpl):
    __dialect__ = 'mysql'
    transactional_ddl = False
    type_synonyms = DefaultImpl.type_synonyms + ({'BOOL', 'TINYINT'}, {'JSON', 'LONGTEXT'})
    type_arg_extract = ['character set ([\\w\\-_]+)', 'collate ([\\w\\-_]+)']

    def alter_column(self, table_name: str, column_name: str, nullable: Optional[bool]=None, server_default: Union[_ServerDefault, Literal[False]]=False, name: Optional[str]=None, type_: Optional[TypeEngine]=None, schema: Optional[str]=None, existing_type: Optional[TypeEngine]=None, existing_server_default: Optional[_ServerDefault]=None, existing_nullable: Optional[bool]=None, autoincrement: Optional[bool]=None, existing_autoincrement: Optional[bool]=None, comment: Optional[Union[str, Literal[False]]]=False, existing_comment: Optional[str]=None, **kw: Any) -> None:
        if sqla_compat._server_default_is_identity(server_default, existing_server_default) or sqla_compat._server_default_is_computed(server_default, existing_server_default):
            super().alter_column(table_name, column_name, nullable=nullable, type_=type_, schema=schema, existing_type=existing_type, existing_nullable=existing_nullable, server_default=server_default, existing_server_default=existing_server_default, **kw)
        if name is not None or self._is_mysql_allowed_functional_default(type_ if type_ is not None else existing_type, server_default):
            self._exec(MySQLChangeColumn(table_name, column_name, schema=schema, newname=name if name is not None else column_name, nullable=nullable if nullable is not None else existing_nullable if existing_nullable is not None else True, type_=type_ if type_ is not None else existing_type, default=server_default if server_default is not False else existing_server_default, autoincrement=autoincrement if autoincrement is not None else existing_autoincrement, comment=comment if comment is not False else existing_comment))
        elif nullable is not None or type_ is not None or autoincrement is not None or (comment is not False):
            self._exec(MySQLModifyColumn(table_name, column_name, schema=schema, newname=name if name is not None else column_name, nullable=nullable if nullable is not None else existing_nullable if existing_nullable is not None else True, type_=type_ if type_ is not None else existing_type, default=server_default if server_default is not False else existing_server_default, autoincrement=autoincrement if autoincrement is not None else existing_autoincrement, comment=comment if comment is not False else existing_comment))
        elif server_default is not False:
            self._exec(MySQLAlterDefault(table_name, column_name, server_default, schema=schema))

    def drop_constraint(self, const: Constraint) -> None:
        if isinstance(const, schema.CheckConstraint) and _is_type_bound(const):
            return
        super().drop_constraint(const)

    def _is_mysql_allowed_functional_default(self, type_: Optional[TypeEngine], server_default: Union[_ServerDefault, Literal[False]]) -> bool:
        return type_ is not None and type_._type_affinity is sqltypes.DateTime and (server_default is not None)

    def compare_server_default(self, inspector_column, metadata_column, rendered_metadata_default, rendered_inspector_default):
        if metadata_column.type._type_affinity is sqltypes.Integer and inspector_column.primary_key and (not inspector_column.autoincrement) and (not rendered_metadata_default) and (rendered_inspector_default == "'0'"):
            return False
        elif rendered_inspector_default and inspector_column.type._type_affinity is sqltypes.Integer:
            rendered_inspector_default = re.sub("^'|'$", '', rendered_inspector_default) if rendered_inspector_default is not None else None
            return rendered_inspector_default != rendered_metadata_default
        elif rendered_metadata_default and metadata_column.type._type_affinity is sqltypes.String:
            metadata_default = re.sub("^'|'$", '', rendered_metadata_default)
            return rendered_inspector_default != f"'{metadata_default}'"
        elif rendered_inspector_default and rendered_metadata_default:
            onupdate_ins = re.match('(.*) (on update.*?)(?:\\(\\))?$', rendered_inspector_default.lower())
            onupdate_met = re.match('(.*) (on update.*?)(?:\\(\\))?$', rendered_metadata_default.lower())
            if onupdate_ins:
                if not onupdate_met:
                    return True
                elif onupdate_ins.group(2) != onupdate_met.group(2):
                    return True
                rendered_inspector_default = onupdate_ins.group(1)
                rendered_metadata_default = onupdate_met.group(1)
            return re.sub('(.*?)(?:\\(\\))?$', '\\1', rendered_inspector_default.lower()) != re.sub('(.*?)(?:\\(\\))?$', '\\1', rendered_metadata_default.lower())
        else:
            return rendered_inspector_default != rendered_metadata_default

    def correct_for_autogen_constraints(self, conn_unique_constraints, conn_indexes, metadata_unique_constraints, metadata_indexes):
        removed = set()
        for idx in list(conn_indexes):
            if idx.unique:
                continue
            for col in idx.columns:
                if idx.name == col.name:
                    conn_indexes.remove(idx)
                    removed.add(idx.name)
                    break
                for fk in col.foreign_keys:
                    if fk.name == idx.name:
                        conn_indexes.remove(idx)
                        removed.add(idx.name)
                        break
                if idx.name in removed:
                    break
        for idx in list(metadata_indexes):
            if idx.name in removed:
                metadata_indexes.remove(idx)

    def correct_for_autogen_foreignkeys(self, conn_fks, metadata_fks):
        conn_fk_by_sig = {self._create_reflected_constraint_sig(fk).unnamed_no_options: fk for fk in conn_fks}
        metadata_fk_by_sig = {self._create_metadata_constraint_sig(fk).unnamed_no_options: fk for fk in metadata_fks}
        for sig in set(conn_fk_by_sig).intersection(metadata_fk_by_sig):
            mdfk = metadata_fk_by_sig[sig]
            cnfk = conn_fk_by_sig[sig]
            if mdfk.ondelete is not None and mdfk.ondelete.lower() == 'restrict' and (cnfk.ondelete is None):
                cnfk.ondelete = 'RESTRICT'
            if mdfk.onupdate is not None and mdfk.onupdate.lower() == 'restrict' and (cnfk.onupdate is None):
                cnfk.onupdate = 'RESTRICT'