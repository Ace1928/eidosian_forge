from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
class MSSQLCompiler(compiler.SQLCompiler):
    returning_precedes_values = True
    extract_map = util.update_copy(compiler.SQLCompiler.extract_map, {'doy': 'dayofyear', 'dow': 'weekday', 'milliseconds': 'millisecond', 'microseconds': 'microsecond'})

    def __init__(self, *args, **kwargs):
        self.tablealiases = {}
        super().__init__(*args, **kwargs)

    def _with_legacy_schema_aliasing(fn):

        def decorate(self, *arg, **kw):
            if self.dialect.legacy_schema_aliasing:
                return fn(self, *arg, **kw)
            else:
                super_ = getattr(super(MSSQLCompiler, self), fn.__name__)
                return super_(*arg, **kw)
        return decorate

    def visit_now_func(self, fn, **kw):
        return 'CURRENT_TIMESTAMP'

    def visit_current_date_func(self, fn, **kw):
        return 'GETDATE()'

    def visit_length_func(self, fn, **kw):
        return 'LEN%s' % self.function_argspec(fn, **kw)

    def visit_char_length_func(self, fn, **kw):
        return 'LEN%s' % self.function_argspec(fn, **kw)

    def visit_aggregate_strings_func(self, fn, **kw):
        expr = fn.clauses.clauses[0]._compiler_dispatch(self, **kw)
        kw['literal_execute'] = True
        delimeter = fn.clauses.clauses[1]._compiler_dispatch(self, **kw)
        return f'string_agg({expr}, {delimeter})'

    def visit_concat_op_expression_clauselist(self, clauselist, operator, **kw):
        return ' + '.join((self.process(elem, **kw) for elem in clauselist))

    def visit_concat_op_binary(self, binary, operator, **kw):
        return '%s + %s' % (self.process(binary.left, **kw), self.process(binary.right, **kw))

    def visit_true(self, expr, **kw):
        return '1'

    def visit_false(self, expr, **kw):
        return '0'

    def visit_match_op_binary(self, binary, operator, **kw):
        return 'CONTAINS (%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))

    def get_select_precolumns(self, select, **kw):
        """MS-SQL puts TOP, it's version of LIMIT here"""
        s = super().get_select_precolumns(select, **kw)
        if select._has_row_limiting_clause and self._use_top(select):
            kw['literal_execute'] = True
            s += 'TOP %s ' % self.process(self._get_limit_or_fetch(select), **kw)
            if select._fetch_clause is not None:
                if select._fetch_clause_options['percent']:
                    s += 'PERCENT '
                if select._fetch_clause_options['with_ties']:
                    s += 'WITH TIES '
        return s

    def get_from_hint_text(self, table, text):
        return text

    def get_crud_hint_text(self, table, text):
        return text

    def _get_limit_or_fetch(self, select):
        if select._fetch_clause is None:
            return select._limit_clause
        else:
            return select._fetch_clause

    def _use_top(self, select):
        return select._offset_clause is None and (select._simple_int_clause(select._limit_clause) or (select._simple_int_clause(select._fetch_clause) and (select._fetch_clause_options['percent'] or select._fetch_clause_options['with_ties'])))

    def limit_clause(self, cs, **kwargs):
        return ''

    def _check_can_use_fetch_limit(self, select):
        if not select._order_by_clause.clauses:
            raise exc.CompileError('MSSQL requires an order_by when using an OFFSET or a non-simple LIMIT clause')
        if select._fetch_clause_options is not None and (select._fetch_clause_options['percent'] or select._fetch_clause_options['with_ties']):
            raise exc.CompileError('MSSQL needs TOP to use PERCENT and/or WITH TIES. Only simple fetch without offset can be used.')

    def _row_limit_clause(self, select, **kw):
        """MSSQL 2012 supports OFFSET/FETCH operators
        Use it instead subquery with row_number

        """
        if self.dialect._supports_offset_fetch and (not self._use_top(select)):
            self._check_can_use_fetch_limit(select)
            return self.fetch_clause(select, fetch_clause=self._get_limit_or_fetch(select), require_offset=True, **kw)
        else:
            return ''

    def visit_try_cast(self, element, **kw):
        return 'TRY_CAST (%s AS %s)' % (self.process(element.clause, **kw), self.process(element.typeclause, **kw))

    def translate_select_structure(self, select_stmt, **kwargs):
        """Look for ``LIMIT`` and OFFSET in a select statement, and if
        so tries to wrap it in a subquery with ``row_number()`` criterion.
        MSSQL 2012 and above are excluded

        """
        select = select_stmt
        if select._has_row_limiting_clause and (not self.dialect._supports_offset_fetch) and (not self._use_top(select)) and (not getattr(select, '_mssql_visit', None)):
            self._check_can_use_fetch_limit(select)
            _order_by_clauses = [sql_util.unwrap_label_reference(elem) for elem in select._order_by_clause.clauses]
            limit_clause = self._get_limit_or_fetch(select)
            offset_clause = select._offset_clause
            select = select._generate()
            select._mssql_visit = True
            select = select.add_columns(sql.func.ROW_NUMBER().over(order_by=_order_by_clauses).label('mssql_rn')).order_by(None).alias()
            mssql_rn = sql.column('mssql_rn')
            limitselect = sql.select(*[c for c in select.c if c.key != 'mssql_rn'])
            if offset_clause is not None:
                limitselect = limitselect.where(mssql_rn > offset_clause)
                if limit_clause is not None:
                    limitselect = limitselect.where(mssql_rn <= limit_clause + offset_clause)
            else:
                limitselect = limitselect.where(mssql_rn <= limit_clause)
            return limitselect
        else:
            return select

    @_with_legacy_schema_aliasing
    def visit_table(self, table, mssql_aliased=False, iscrud=False, **kwargs):
        if mssql_aliased is table or iscrud:
            return super().visit_table(table, **kwargs)
        alias = self._schema_aliased_table(table)
        if alias is not None:
            return self.process(alias, mssql_aliased=table, **kwargs)
        else:
            return super().visit_table(table, **kwargs)

    @_with_legacy_schema_aliasing
    def visit_alias(self, alias, **kw):
        kw['mssql_aliased'] = alias.element
        return super().visit_alias(alias, **kw)

    @_with_legacy_schema_aliasing
    def visit_column(self, column, add_to_result_map=None, **kw):
        if column.table is not None and (not self.isupdate and (not self.isdelete)) or self.is_subquery():
            t = self._schema_aliased_table(column.table)
            if t is not None:
                converted = elements._corresponding_column_or_error(t, column)
                if add_to_result_map is not None:
                    add_to_result_map(column.name, column.name, (column, column.name, column.key), column.type)
                return super().visit_column(converted, **kw)
        return super().visit_column(column, add_to_result_map=add_to_result_map, **kw)

    def _schema_aliased_table(self, table):
        if getattr(table, 'schema', None) is not None:
            if table not in self.tablealiases:
                self.tablealiases[table] = table.alias()
            return self.tablealiases[table]
        else:
            return None

    def visit_extract(self, extract, **kw):
        field = self.extract_map.get(extract.field, extract.field)
        return 'DATEPART(%s, %s)' % (field, self.process(extract.expr, **kw))

    def visit_savepoint(self, savepoint_stmt, **kw):
        return 'SAVE TRANSACTION %s' % self.preparer.format_savepoint(savepoint_stmt)

    def visit_rollback_to_savepoint(self, savepoint_stmt, **kw):
        return 'ROLLBACK TRANSACTION %s' % self.preparer.format_savepoint(savepoint_stmt)

    def visit_binary(self, binary, **kwargs):
        """Move bind parameters to the right-hand side of an operator, where
        possible.

        """
        if isinstance(binary.left, expression.BindParameter) and binary.operator == operator.eq and (not isinstance(binary.right, expression.BindParameter)):
            return self.process(expression.BinaryExpression(binary.right, binary.left, binary.operator), **kwargs)
        return super().visit_binary(binary, **kwargs)

    def returning_clause(self, stmt, returning_cols, *, populate_result_map, **kw):
        if stmt.is_insert or stmt.is_update:
            target = stmt.table.alias('inserted')
        elif stmt.is_delete:
            target = stmt.table.alias('deleted')
        else:
            assert False, 'expected Insert, Update or Delete statement'
        adapter = sql_util.ClauseAdapter(target)
        columns = [self._label_returning_column(stmt, adapter.traverse(column), populate_result_map, {'result_map_targets': (column,)}, fallback_label_name=fallback_label_name, column_is_repeated=repeated, name=name, proxy_name=proxy_name, **kw) for name, proxy_name, fallback_label_name, column, repeated in stmt._generate_columns_plus_names(True, cols=expression._select_iterables(returning_cols))]
        return 'OUTPUT ' + ', '.join(columns)

    def get_cte_preamble(self, recursive):
        return 'WITH'

    def label_select_column(self, select, column, asfrom):
        if isinstance(column, expression.Function):
            return column.label(None)
        else:
            return super().label_select_column(select, column, asfrom)

    def for_update_clause(self, select, **kw):
        return ''

    def order_by_clause(self, select, **kw):
        if self.is_subquery() and (not self._use_top(select)) and (select._offset is None or not self.dialect._supports_offset_fetch):
            return ''
        order_by = self.process(select._order_by_clause, **kw)
        if order_by:
            return ' ORDER BY ' + order_by
        else:
            return ''

    def update_from_clause(self, update_stmt, from_table, extra_froms, from_hints, **kw):
        """Render the UPDATE..FROM clause specific to MSSQL.

        In MSSQL, if the UPDATE statement involves an alias of the table to
        be updated, then the table itself must be added to the FROM list as
        well. Otherwise, it is optional. Here, we add it regardless.

        """
        return 'FROM ' + ', '.join((t._compiler_dispatch(self, asfrom=True, fromhints=from_hints, **kw) for t in [from_table] + extra_froms))

    def delete_table_clause(self, delete_stmt, from_table, extra_froms, **kw):
        """If we have extra froms make sure we render any alias as hint."""
        ashint = False
        if extra_froms:
            ashint = True
        return from_table._compiler_dispatch(self, asfrom=True, iscrud=True, ashint=ashint, **kw)

    def delete_extra_from_clause(self, delete_stmt, from_table, extra_froms, from_hints, **kw):
        """Render the DELETE .. FROM clause specific to MSSQL.

        Yes, it has the FROM keyword twice.

        """
        return 'FROM ' + ', '.join((t._compiler_dispatch(self, asfrom=True, fromhints=from_hints, **kw) for t in [from_table] + extra_froms))

    def visit_empty_set_expr(self, type_, **kw):
        return 'SELECT 1 WHERE 1!=1'

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return 'NOT EXISTS (SELECT %s INTERSECT SELECT %s)' % (self.process(binary.left), self.process(binary.right))

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return 'EXISTS (SELECT %s INTERSECT SELECT %s)' % (self.process(binary.left), self.process(binary.right))

    def _render_json_extract_from_binary(self, binary, operator, **kw):
        if binary.type._type_affinity is sqltypes.JSON:
            return 'JSON_QUERY(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        case_expression = 'CASE JSON_VALUE(%s, %s) WHEN NULL THEN NULL' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        if binary.type._type_affinity is sqltypes.Integer:
            type_expression = 'ELSE CAST(JSON_VALUE(%s, %s) AS INTEGER)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        elif binary.type._type_affinity is sqltypes.Numeric:
            type_expression = 'ELSE CAST(JSON_VALUE(%s, %s) AS %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw), 'FLOAT' if isinstance(binary.type, sqltypes.Float) else 'NUMERIC(%s, %s)' % (binary.type.precision, binary.type.scale))
        elif binary.type._type_affinity is sqltypes.Boolean:
            type_expression = "WHEN 'true' THEN 1 WHEN 'false' THEN 0 ELSE NULL"
        elif binary.type._type_affinity is sqltypes.String:
            type_expression = 'ELSE JSON_VALUE(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        else:
            type_expression = 'ELSE JSON_QUERY(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        return case_expression + ' ' + type_expression + ' END'

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        return self._render_json_extract_from_binary(binary, operator, **kw)

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        return self._render_json_extract_from_binary(binary, operator, **kw)

    def visit_sequence(self, seq, **kw):
        return 'NEXT VALUE FOR %s' % self.preparer.format_sequence(seq)