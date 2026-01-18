from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
class OracleCompiler(compiler.SQLCompiler):
    """Oracle compiler modifies the lexical structure of Select
    statements to work under non-ANSI configured Oracle databases, if
    the use_ansi flag is False.
    """
    compound_keywords = util.update_copy(compiler.SQLCompiler.compound_keywords, {expression.CompoundSelect.EXCEPT: 'MINUS'})

    def __init__(self, *args, **kwargs):
        self.__wheres = {}
        super().__init__(*args, **kwargs)

    def visit_mod_binary(self, binary, operator, **kw):
        return 'mod(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))

    def visit_now_func(self, fn, **kw):
        return 'CURRENT_TIMESTAMP'

    def visit_char_length_func(self, fn, **kw):
        return 'LENGTH' + self.function_argspec(fn, **kw)

    def visit_match_op_binary(self, binary, operator, **kw):
        return 'CONTAINS (%s, %s)' % (self.process(binary.left), self.process(binary.right))

    def visit_true(self, expr, **kw):
        return '1'

    def visit_false(self, expr, **kw):
        return '0'

    def get_cte_preamble(self, recursive):
        return 'WITH'

    def get_select_hint_text(self, byfroms):
        return ' '.join(('/*+ %s */' % text for table, text in byfroms.items()))

    def function_argspec(self, fn, **kw):
        if len(fn.clauses) > 0 or fn.name.upper() not in NO_ARG_FNS:
            return compiler.SQLCompiler.function_argspec(self, fn, **kw)
        else:
            return ''

    def visit_function(self, func, **kw):
        text = super().visit_function(func, **kw)
        if kw.get('asfrom', False):
            text = 'TABLE (%s)' % text
        return text

    def visit_table_valued_column(self, element, **kw):
        text = super().visit_table_valued_column(element, **kw)
        text = text + '.COLUMN_VALUE'
        return text

    def default_from(self):
        """Called when a ``SELECT`` statement has no froms,
        and no ``FROM`` clause is to be appended.

        The Oracle compiler tacks a "FROM DUAL" to the statement.
        """
        return ' FROM DUAL'

    def visit_join(self, join, from_linter=None, **kwargs):
        if self.dialect.use_ansi:
            return compiler.SQLCompiler.visit_join(self, join, from_linter=from_linter, **kwargs)
        else:
            if from_linter:
                from_linter.edges.add((join.left, join.right))
            kwargs['asfrom'] = True
            if isinstance(join.right, expression.FromGrouping):
                right = join.right.element
            else:
                right = join.right
            return self.process(join.left, from_linter=from_linter, **kwargs) + ', ' + self.process(right, from_linter=from_linter, **kwargs)

    def _get_nonansi_join_whereclause(self, froms):
        clauses = []

        def visit_join(join):
            if join.isouter:

                def visit_binary(binary):
                    if isinstance(binary.left, expression.ColumnClause) and join.right.is_derived_from(binary.left.table):
                        binary.left = _OuterJoinColumn(binary.left)
                    elif isinstance(binary.right, expression.ColumnClause) and join.right.is_derived_from(binary.right.table):
                        binary.right = _OuterJoinColumn(binary.right)
                clauses.append(visitors.cloned_traverse(join.onclause, {}, {'binary': visit_binary}))
            else:
                clauses.append(join.onclause)
            for j in (join.left, join.right):
                if isinstance(j, expression.Join):
                    visit_join(j)
                elif isinstance(j, expression.FromGrouping):
                    visit_join(j.element)
        for f in froms:
            if isinstance(f, expression.Join):
                visit_join(f)
        if not clauses:
            return None
        else:
            return sql.and_(*clauses)

    def visit_outer_join_column(self, vc, **kw):
        return self.process(vc.column, **kw) + '(+)'

    def visit_sequence(self, seq, **kw):
        return self.preparer.format_sequence(seq) + '.nextval'

    def get_render_as_alias_suffix(self, alias_name_text):
        """Oracle doesn't like ``FROM table AS alias``"""
        return ' ' + alias_name_text

    def returning_clause(self, stmt, returning_cols, *, populate_result_map, **kw):
        columns = []
        binds = []
        for i, column in enumerate(expression._select_iterables(returning_cols)):
            if self.isupdate and isinstance(column, sa_schema.Column) and isinstance(column.server_default, Computed) and (not self.dialect._supports_update_returning_computed_cols):
                util.warn("Computed columns don't work with Oracle UPDATE statements that use RETURNING; the value of the column *before* the UPDATE takes place is returned.   It is advised to not use RETURNING with an Oracle computed column.  Consider setting implicit_returning to False on the Table object in order to avoid implicit RETURNING clauses from being generated for this Table.")
            if column.type._has_column_expression:
                col_expr = column.type.column_expression(column)
            else:
                col_expr = column
            outparam = sql.outparam('ret_%d' % i, type_=column.type)
            self.binds[outparam.key] = outparam
            binds.append(self.bindparam_string(self._truncate_bindparam(outparam)))
            if self.has_out_parameters:
                raise exc.InvalidRequestError('Using explicit outparam() objects with UpdateBase.returning() in the same Core DML statement is not supported in the Oracle dialect.')
            self._oracle_returning = True
            columns.append(self.process(col_expr, within_columns_clause=False))
            if populate_result_map:
                self._add_to_result_map(getattr(col_expr, 'name', col_expr._anon_name_label), getattr(col_expr, 'name', col_expr._anon_name_label), (column, getattr(column, 'name', None), getattr(column, 'key', None)), column.type)
        return 'RETURNING ' + ', '.join(columns) + ' INTO ' + ', '.join(binds)

    def _row_limit_clause(self, select, **kw):
        """ORacle 12c supports OFFSET/FETCH operators
        Use it instead subquery with row_number

        """
        if select._fetch_clause is not None or not self.dialect._supports_offset_fetch:
            return super()._row_limit_clause(select, use_literal_execute_for_simple_int=True, **kw)
        else:
            return self.fetch_clause(select, fetch_clause=self._get_limit_or_fetch(select), use_literal_execute_for_simple_int=True, **kw)

    def _get_limit_or_fetch(self, select):
        if select._fetch_clause is None:
            return select._limit_clause
        else:
            return select._fetch_clause

    def translate_select_structure(self, select_stmt, **kwargs):
        select = select_stmt
        if not getattr(select, '_oracle_visit', None):
            if not self.dialect.use_ansi:
                froms = self._display_froms_for_select(select, kwargs.get('asfrom', False))
                whereclause = self._get_nonansi_join_whereclause(froms)
                if whereclause is not None:
                    select = select.where(whereclause)
                    select._oracle_visit = True
            if select._has_row_limiting_clause and (not self.dialect._supports_offset_fetch) and (select._fetch_clause is None):
                limit_clause = select._limit_clause
                offset_clause = select._offset_clause
                if select._simple_int_clause(limit_clause):
                    limit_clause = limit_clause.render_literal_execute()
                if select._simple_int_clause(offset_clause):
                    offset_clause = offset_clause.render_literal_execute()
                orig_select = select
                select = select._generate()
                select._oracle_visit = True
                for_update = select._for_update_arg
                if for_update is not None and for_update.of:
                    for_update = for_update._clone()
                    for_update._copy_internals()
                    for elem in for_update.of:
                        if not select.selected_columns.contains_column(elem):
                            select = select.add_columns(elem)
                inner_subquery = select.alias()
                limitselect = sql.select(*[c for c in inner_subquery.c if orig_select.selected_columns.corresponding_column(c) is not None])
                if limit_clause is not None and self.dialect.optimize_limits and select._simple_int_clause(limit_clause):
                    limitselect = limitselect.prefix_with(expression.text('/*+ FIRST_ROWS(%s) */' % self.process(limit_clause, **kwargs)))
                limitselect._oracle_visit = True
                limitselect._is_wrapper = True
                if for_update is not None and for_update.of:
                    adapter = sql_util.ClauseAdapter(inner_subquery)
                    for_update.of = [adapter.traverse(elem) for elem in for_update.of]
                if limit_clause is not None:
                    if select._simple_int_clause(limit_clause) and (offset_clause is None or select._simple_int_clause(offset_clause)):
                        max_row = limit_clause
                        if offset_clause is not None:
                            max_row = max_row + offset_clause
                    else:
                        max_row = limit_clause
                        if offset_clause is not None:
                            max_row = max_row + offset_clause
                    limitselect = limitselect.where(sql.literal_column('ROWNUM') <= max_row)
                if offset_clause is None:
                    limitselect._for_update_arg = for_update
                    select = limitselect
                else:
                    limitselect = limitselect.add_columns(sql.literal_column('ROWNUM').label('ora_rn'))
                    limitselect._oracle_visit = True
                    limitselect._is_wrapper = True
                    if for_update is not None and for_update.of:
                        limitselect_cols = limitselect.selected_columns
                        for elem in for_update.of:
                            if limitselect_cols.corresponding_column(elem) is None:
                                limitselect = limitselect.add_columns(elem)
                    limit_subquery = limitselect.alias()
                    origselect_cols = orig_select.selected_columns
                    offsetselect = sql.select(*[c for c in limit_subquery.c if origselect_cols.corresponding_column(c) is not None])
                    offsetselect._oracle_visit = True
                    offsetselect._is_wrapper = True
                    if for_update is not None and for_update.of:
                        adapter = sql_util.ClauseAdapter(limit_subquery)
                        for_update.of = [adapter.traverse(elem) for elem in for_update.of]
                    offsetselect = offsetselect.where(sql.literal_column('ora_rn') > offset_clause)
                    offsetselect._for_update_arg = for_update
                    select = offsetselect
        return select

    def limit_clause(self, select, **kw):
        return ''

    def visit_empty_set_expr(self, type_, **kw):
        return 'SELECT 1 FROM DUAL WHERE 1!=1'

    def for_update_clause(self, select, **kw):
        if self.is_subquery():
            return ''
        tmp = ' FOR UPDATE'
        if select._for_update_arg.of:
            tmp += ' OF ' + ', '.join((self.process(elem, **kw) for elem in select._for_update_arg.of))
        if select._for_update_arg.nowait:
            tmp += ' NOWAIT'
        if select._for_update_arg.skip_locked:
            tmp += ' SKIP LOCKED'
        return tmp

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return 'DECODE(%s, %s, 0, 1) = 1' % (self.process(binary.left), self.process(binary.right))

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return 'DECODE(%s, %s, 0, 1) = 0' % (self.process(binary.left), self.process(binary.right))

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        string = self.process(binary.left, **kw)
        pattern = self.process(binary.right, **kw)
        flags = binary.modifiers['flags']
        if flags is None:
            return 'REGEXP_LIKE(%s, %s)' % (string, pattern)
        else:
            return 'REGEXP_LIKE(%s, %s, %s)' % (string, pattern, self.render_literal_value(flags, sqltypes.STRINGTYPE))

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return 'NOT %s' % self.visit_regexp_match_op_binary(binary, operator, **kw)

    def visit_regexp_replace_op_binary(self, binary, operator, **kw):
        string = self.process(binary.left, **kw)
        pattern_replace = self.process(binary.right, **kw)
        flags = binary.modifiers['flags']
        if flags is None:
            return 'REGEXP_REPLACE(%s, %s)' % (string, pattern_replace)
        else:
            return 'REGEXP_REPLACE(%s, %s, %s)' % (string, pattern_replace, self.render_literal_value(flags, sqltypes.STRINGTYPE))

    def visit_aggregate_strings_func(self, fn, **kw):
        return 'LISTAGG%s' % self.function_argspec(fn, **kw)