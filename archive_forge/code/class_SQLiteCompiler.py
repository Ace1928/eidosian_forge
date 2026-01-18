from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
class SQLiteCompiler(compiler.SQLCompiler):
    extract_map = util.update_copy(compiler.SQLCompiler.extract_map, {'month': '%m', 'day': '%d', 'year': '%Y', 'second': '%S', 'hour': '%H', 'doy': '%j', 'minute': '%M', 'epoch': '%s', 'dow': '%w', 'week': '%W'})

    def visit_truediv_binary(self, binary, operator, **kw):
        return self.process(binary.left, **kw) + ' / ' + '(%s + 0.0)' % self.process(binary.right, **kw)

    def visit_now_func(self, fn, **kw):
        return 'CURRENT_TIMESTAMP'

    def visit_localtimestamp_func(self, func, **kw):
        return 'DATETIME(CURRENT_TIMESTAMP, "localtime")'

    def visit_true(self, expr, **kw):
        return '1'

    def visit_false(self, expr, **kw):
        return '0'

    def visit_char_length_func(self, fn, **kw):
        return 'length%s' % self.function_argspec(fn)

    def visit_aggregate_strings_func(self, fn, **kw):
        return 'group_concat%s' % self.function_argspec(fn)

    def visit_cast(self, cast, **kwargs):
        if self.dialect.supports_cast:
            return super().visit_cast(cast, **kwargs)
        else:
            return self.process(cast.clause, **kwargs)

    def visit_extract(self, extract, **kw):
        try:
            return "CAST(STRFTIME('%s', %s) AS INTEGER)" % (self.extract_map[extract.field], self.process(extract.expr, **kw))
        except KeyError as err:
            raise exc.CompileError('%s is not a valid extract argument.' % extract.field) from err

    def returning_clause(self, stmt, returning_cols, *, populate_result_map, **kw):
        kw['include_table'] = False
        return super().returning_clause(stmt, returning_cols, populate_result_map=populate_result_map, **kw)

    def limit_clause(self, select, **kw):
        text = ''
        if select._limit_clause is not None:
            text += '\n LIMIT ' + self.process(select._limit_clause, **kw)
        if select._offset_clause is not None:
            if select._limit_clause is None:
                text += '\n LIMIT ' + self.process(sql.literal(-1))
            text += ' OFFSET ' + self.process(select._offset_clause, **kw)
        else:
            text += ' OFFSET ' + self.process(sql.literal(0), **kw)
        return text

    def for_update_clause(self, select, **kw):
        return ''

    def update_from_clause(self, update_stmt, from_table, extra_froms, from_hints, **kw):
        kw['asfrom'] = True
        return 'FROM ' + ', '.join((t._compiler_dispatch(self, fromhints=from_hints, **kw) for t in extra_froms))

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return '%s IS NOT %s' % (self.process(binary.left), self.process(binary.right))

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return '%s IS %s' % (self.process(binary.left), self.process(binary.right))

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        if binary.type._type_affinity is sqltypes.JSON:
            expr = 'JSON_QUOTE(JSON_EXTRACT(%s, %s))'
        else:
            expr = 'JSON_EXTRACT(%s, %s)'
        return expr % (self.process(binary.left, **kw), self.process(binary.right, **kw))

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        if binary.type._type_affinity is sqltypes.JSON:
            expr = 'JSON_QUOTE(JSON_EXTRACT(%s, %s))'
        else:
            expr = 'JSON_EXTRACT(%s, %s)'
        return expr % (self.process(binary.left, **kw), self.process(binary.right, **kw))

    def visit_empty_set_op_expr(self, type_, expand_op, **kw):
        return self.visit_empty_set_expr(type_)

    def visit_empty_set_expr(self, element_types, **kw):
        return 'SELECT %s FROM (SELECT %s) WHERE 1!=1' % (', '.join(('1' for type_ in element_types or [INTEGER()])), ', '.join(('1' for type_ in element_types or [INTEGER()])))

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, ' REGEXP ', **kw)

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, ' NOT REGEXP ', **kw)

    def _on_conflict_target(self, clause, **kw):
        if clause.constraint_target is not None:
            target_text = '(%s)' % clause.constraint_target
        elif clause.inferred_target_elements is not None:
            target_text = '(%s)' % ', '.join((self.preparer.quote(c) if isinstance(c, str) else self.process(c, include_table=False, use_schema=False) for c in clause.inferred_target_elements))
            if clause.inferred_target_whereclause is not None:
                target_text += ' WHERE %s' % self.process(clause.inferred_target_whereclause, include_table=False, use_schema=False, literal_binds=True)
        else:
            target_text = ''
        return target_text

    def visit_on_conflict_do_nothing(self, on_conflict, **kw):
        target_text = self._on_conflict_target(on_conflict, **kw)
        if target_text:
            return 'ON CONFLICT %s DO NOTHING' % target_text
        else:
            return 'ON CONFLICT DO NOTHING'

    def visit_on_conflict_do_update(self, on_conflict, **kw):
        clause = on_conflict
        target_text = self._on_conflict_target(on_conflict, **kw)
        action_set_ops = []
        set_parameters = dict(clause.update_values_to_set)
        insert_statement = self.stack[-1]['selectable']
        cols = insert_statement.table.c
        for c in cols:
            col_key = c.key
            if col_key in set_parameters:
                value = set_parameters.pop(col_key)
            elif c in set_parameters:
                value = set_parameters.pop(c)
            else:
                continue
            if coercions._is_literal(value):
                value = elements.BindParameter(None, value, type_=c.type)
            elif isinstance(value, elements.BindParameter) and value.type._isnull:
                value = value._clone()
                value.type = c.type
            value_text = self.process(value.self_group(), use_schema=False)
            key_text = self.preparer.quote(c.name)
            action_set_ops.append('%s = %s' % (key_text, value_text))
        if set_parameters:
            util.warn("Additional column names not matching any column keys in table '%s': %s" % (self.current_executable.table.name, ', '.join(("'%s'" % c for c in set_parameters))))
            for k, v in set_parameters.items():
                key_text = self.preparer.quote(k) if isinstance(k, str) else self.process(k, use_schema=False)
                value_text = self.process(coercions.expect(roles.ExpressionElementRole, v), use_schema=False)
                action_set_ops.append('%s = %s' % (key_text, value_text))
        action_text = ', '.join(action_set_ops)
        if clause.update_whereclause is not None:
            action_text += ' WHERE %s' % self.process(clause.update_whereclause, include_table=True, use_schema=False)
        return 'ON CONFLICT %s DO UPDATE SET %s' % (target_text, action_text)