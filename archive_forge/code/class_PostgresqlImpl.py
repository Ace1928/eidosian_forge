from __future__ import annotations
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import Numeric
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import NULLTYPE
from .base import alter_column
from .base import alter_table
from .base import AlterColumn
from .base import ColumnComment
from .base import format_column_name
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import ComparisonResult
from .impl import DefaultImpl
from .. import util
from ..autogenerate import render
from ..operations import ops
from ..operations import schemaobj
from ..operations.base import BatchOperations
from ..operations.base import Operations
from ..util import sqla_compat
from ..util.sqla_compat import compiles
class PostgresqlImpl(DefaultImpl):
    __dialect__ = 'postgresql'
    transactional_ddl = True
    type_synonyms = DefaultImpl.type_synonyms + ({'FLOAT', 'DOUBLE PRECISION'},)

    def create_index(self, index: Index, **kw: Any) -> None:
        postgresql_include = index.kwargs.get('postgresql_include', None) or ()
        for col in postgresql_include:
            if col not in index.table.c:
                index.table.append_column(Column(col, sqltypes.NullType))
        self._exec(CreateIndex(index, **kw))

    def prep_table_for_batch(self, batch_impl, table):
        for constraint in table.constraints:
            if constraint.name is not None and constraint.name in batch_impl.named_constraints:
                self.drop_constraint(constraint)

    def compare_server_default(self, inspector_column, metadata_column, rendered_metadata_default, rendered_inspector_default):
        if metadata_column.primary_key and metadata_column is metadata_column.table._autoincrement_column:
            return False
        conn_col_default = rendered_inspector_default
        defaults_equal = conn_col_default == rendered_metadata_default
        if defaults_equal:
            return False
        if None in (conn_col_default, rendered_metadata_default, metadata_column.server_default):
            return not defaults_equal
        metadata_default = metadata_column.server_default.arg
        if isinstance(metadata_default, str):
            if not isinstance(inspector_column.type, Numeric):
                metadata_default = re.sub("^'|'$", '', metadata_default)
                metadata_default = f"'{metadata_default}'"
            metadata_default = literal_column(metadata_default)
        conn = self.connection
        assert conn is not None
        return not conn.scalar(sqla_compat._select(literal_column(conn_col_default) == metadata_default))

    def alter_column(self, table_name: str, column_name: str, nullable: Optional[bool]=None, server_default: Union[_ServerDefault, Literal[False]]=False, name: Optional[str]=None, type_: Optional[TypeEngine]=None, schema: Optional[str]=None, autoincrement: Optional[bool]=None, existing_type: Optional[TypeEngine]=None, existing_server_default: Optional[_ServerDefault]=None, existing_nullable: Optional[bool]=None, existing_autoincrement: Optional[bool]=None, **kw: Any) -> None:
        using = kw.pop('postgresql_using', None)
        if using is not None and type_ is None:
            raise util.CommandError('postgresql_using must be used with the type_ parameter')
        if type_ is not None:
            self._exec(PostgresqlColumnType(table_name, column_name, type_, schema=schema, using=using, existing_type=existing_type, existing_server_default=existing_server_default, existing_nullable=existing_nullable))
        super().alter_column(table_name, column_name, nullable=nullable, server_default=server_default, name=name, schema=schema, autoincrement=autoincrement, existing_type=existing_type, existing_server_default=existing_server_default, existing_nullable=existing_nullable, existing_autoincrement=existing_autoincrement, **kw)

    def autogen_column_reflect(self, inspector, table, column_info):
        if column_info.get('default') and isinstance(column_info['type'], (INTEGER, BIGINT)):
            seq_match = re.match("nextval\\('(.+?)'::regclass\\)", column_info['default'])
            if seq_match:
                info = sqla_compat._exec_on_inspector(inspector, text("select c.relname, a.attname from pg_class as c join pg_depend d on d.objid=c.oid and d.classid='pg_class'::regclass and d.refclassid='pg_class'::regclass join pg_class t on t.oid=d.refobjid join pg_attribute a on a.attrelid=t.oid and a.attnum=d.refobjsubid where c.relkind='S' and c.relname=:seqname"), seqname=seq_match.group(1)).first()
                if info:
                    seqname, colname = info
                    if colname == column_info['name']:
                        log.info("Detected sequence named '%s' as owned by integer column '%s(%s)', assuming SERIAL and omitting", seqname, table.name, colname)
                        del column_info['default']

    def correct_for_autogen_constraints(self, conn_unique_constraints, conn_indexes, metadata_unique_constraints, metadata_indexes):
        doubled_constraints = {index for index in conn_indexes if index.info.get('duplicates_constraint')}
        for ix in doubled_constraints:
            conn_indexes.remove(ix)
        if not sqla_compat.sqla_2:
            self._skip_functional_indexes(metadata_indexes, conn_indexes)
    _default_modifiers_re = (re.compile('( asc nulls last)$'), re.compile('(?<! desc)( nulls last)$'), re.compile('( asc)$'), re.compile('( asc) nulls first$'), re.compile(' desc( nulls first)$'))

    def _cleanup_index_expr(self, index: Index, expr: str) -> str:
        expr = expr.lower().replace('"', '').replace("'", '')
        if index.table is not None:
            expr = expr.replace(f'{index.table.name.lower()}.', '')
        if '::' in expr:
            expr = re.sub('(::[\\w ]+\\w)', '', expr)
        while expr and expr[0] == '(' and (expr[-1] == ')'):
            expr = expr[1:-1]
        for rs in self._default_modifiers_re:
            if (match := rs.search(expr)):
                start, end = match.span(1)
                expr = expr[:start] + expr[end:]
                break
        while expr and expr[0] == '(' and (expr[-1] == ')'):
            expr = expr[1:-1]
        cast_re = re.compile('cast\\s*\\(')
        if cast_re.match(expr):
            expr = cast_re.sub('', expr)
            expr = re.sub('as\\s+[^)]+\\)', '', expr)
        expr = expr.replace(' ', '')
        return expr

    def _dialect_options(self, item: Union[Index, UniqueConstraint]) -> Tuple[Any, ...]:
        if item.dialect_kwargs.get('postgresql_nulls_not_distinct'):
            return ('nulls_not_distinct',)
        return ()

    def compare_indexes(self, metadata_index: Index, reflected_index: Index) -> ComparisonResult:
        msg = []
        unique_msg = self._compare_index_unique(metadata_index, reflected_index)
        if unique_msg:
            msg.append(unique_msg)
        m_exprs = metadata_index.expressions
        r_exprs = reflected_index.expressions
        if len(m_exprs) != len(r_exprs):
            msg.append(f'expression number {len(r_exprs)} to {len(m_exprs)}')
        if msg:
            return ComparisonResult.Different(msg)
        skip = []
        for pos, (m_e, r_e) in enumerate(zip(m_exprs, r_exprs), 1):
            m_compile = self._compile_element(m_e)
            m_text = self._cleanup_index_expr(metadata_index, m_compile)
            r_compile = self._compile_element(r_e)
            r_text = self._cleanup_index_expr(metadata_index, r_compile)
            if m_text == r_text:
                continue
            elif m_compile.strip().endswith('_ops') and (' ' in m_compile or ')' in m_compile):
                skip.append(f'expression #{pos} {m_compile!r} detected as including operator clause.')
                util.warn(f'Expression #{pos} {m_compile!r} in index {reflected_index.name!r} detected to include an operator clause. Expression compare cannot proceed. Please move the operator clause to the ``postgresql_ops`` dict to enable proper compare of the index expressions: https://docs.sqlalchemy.org/en/latest/dialects/postgresql.html#operator-classes')
            else:
                msg.append(f'expression #{pos} {r_compile!r} to {m_compile!r}')
        m_options = self._dialect_options(metadata_index)
        r_options = self._dialect_options(reflected_index)
        if m_options != r_options:
            msg.extend(f'options {r_options} to {m_options}')
        if msg:
            return ComparisonResult.Different(msg)
        elif skip:
            return ComparisonResult.Skip(skip)
        else:
            return ComparisonResult.Equal()

    def compare_unique_constraint(self, metadata_constraint: UniqueConstraint, reflected_constraint: UniqueConstraint) -> ComparisonResult:
        metadata_tup = self._create_metadata_constraint_sig(metadata_constraint)
        reflected_tup = self._create_reflected_constraint_sig(reflected_constraint)
        meta_sig = metadata_tup.unnamed
        conn_sig = reflected_tup.unnamed
        if conn_sig != meta_sig:
            return ComparisonResult.Different(f'expression {conn_sig} to {meta_sig}')
        metadata_do = self._dialect_options(metadata_tup.const)
        conn_do = self._dialect_options(reflected_tup.const)
        if metadata_do != conn_do:
            return ComparisonResult.Different(f'expression {conn_do} to {metadata_do}')
        return ComparisonResult.Equal()

    def adjust_reflected_dialect_options(self, reflected_options: Dict[str, Any], kind: str) -> Dict[str, Any]:
        options: Dict[str, Any]
        options = reflected_options.get('dialect_options', {}).copy()
        if not options.get('postgresql_include'):
            options.pop('postgresql_include', None)
        return options

    def _compile_element(self, element: Union[ClauseElement, str]) -> str:
        if isinstance(element, str):
            return element
        return element.compile(dialect=self.dialect, compile_kwargs={'literal_binds': True, 'include_table': False}).string

    def render_ddl_sql_expr(self, expr: ClauseElement, is_server_default: bool=False, is_index: bool=False, **kw: Any) -> str:
        """Render a SQL expression that is typically a server default,
        index expression, etc.

        """
        if is_index and (not isinstance(expr, ColumnClause)):
            expr = expr.self_group()
        return super().render_ddl_sql_expr(expr, is_server_default=is_server_default, is_index=is_index, **kw)

    def render_type(self, type_: TypeEngine, autogen_context: AutogenContext) -> Union[str, Literal[False]]:
        mod = type(type_).__module__
        if not mod.startswith('sqlalchemy.dialects.postgresql'):
            return False
        if hasattr(self, '_render_%s_type' % type_.__visit_name__):
            meth = getattr(self, '_render_%s_type' % type_.__visit_name__)
            return meth(type_, autogen_context)
        return False

    def _render_HSTORE_type(self, type_: HSTORE, autogen_context: AutogenContext) -> str:
        return cast(str, render._render_type_w_subtype(type_, autogen_context, 'text_type', '(.+?\\(.*text_type=)'))

    def _render_ARRAY_type(self, type_: ARRAY, autogen_context: AutogenContext) -> str:
        return cast(str, render._render_type_w_subtype(type_, autogen_context, 'item_type', '(.+?\\()'))

    def _render_JSON_type(self, type_: JSON, autogen_context: AutogenContext) -> str:
        return cast(str, render._render_type_w_subtype(type_, autogen_context, 'astext_type', '(.+?\\(.*astext_type=)'))

    def _render_JSONB_type(self, type_: JSONB, autogen_context: AutogenContext) -> str:
        return cast(str, render._render_type_w_subtype(type_, autogen_context, 'astext_type', '(.+?\\(.*astext_type=)'))