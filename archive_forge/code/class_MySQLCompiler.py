from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
class MySQLCompiler(compiler.SQLCompiler):
    render_table_with_column_in_update_from = True
    'Overridden from base SQLCompiler value'
    extract_map = compiler.SQLCompiler.extract_map.copy()
    extract_map.update({'milliseconds': 'millisecond'})

    def default_from(self):
        """Called when a ``SELECT`` statement has no froms,
        and no ``FROM`` clause is to be appended.

        """
        if self.stack:
            stmt = self.stack[-1]['selectable']
            if stmt._where_criteria:
                return ' FROM DUAL'
        return ''

    def visit_random_func(self, fn, **kw):
        return 'rand%s' % self.function_argspec(fn)

    def visit_rollup_func(self, fn, **kw):
        clause = ', '.join((elem._compiler_dispatch(self, **kw) for elem in fn.clauses))
        return f'{clause} WITH ROLLUP'

    def visit_aggregate_strings_func(self, fn, **kw):
        expr, delimeter = (elem._compiler_dispatch(self, **kw) for elem in fn.clauses)
        return f'group_concat({expr} SEPARATOR {delimeter})'

    def visit_sequence(self, seq, **kw):
        return 'nextval(%s)' % self.preparer.format_sequence(seq)

    def visit_sysdate_func(self, fn, **kw):
        return 'SYSDATE()'

    def _render_json_extract_from_binary(self, binary, operator, **kw):
        if binary.type._type_affinity is sqltypes.JSON:
            return 'JSON_EXTRACT(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        case_expression = "CASE JSON_EXTRACT(%s, %s) WHEN 'null' THEN NULL" % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        if binary.type._type_affinity is sqltypes.Integer:
            type_expression = 'ELSE CAST(JSON_EXTRACT(%s, %s) AS SIGNED INTEGER)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        elif binary.type._type_affinity is sqltypes.Numeric:
            if binary.type.scale is not None and binary.type.precision is not None:
                type_expression = 'ELSE CAST(JSON_EXTRACT(%s, %s) AS DECIMAL(%s, %s))' % (self.process(binary.left, **kw), self.process(binary.right, **kw), binary.type.precision, binary.type.scale)
            else:
                type_expression = 'ELSE JSON_EXTRACT(%s, %s)+0.0000000000000000000000' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        elif binary.type._type_affinity is sqltypes.Boolean:
            type_expression = 'WHEN true THEN true ELSE false'
        elif binary.type._type_affinity is sqltypes.String:
            type_expression = 'ELSE JSON_UNQUOTE(JSON_EXTRACT(%s, %s))' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        else:
            type_expression = 'ELSE JSON_EXTRACT(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        return case_expression + ' ' + type_expression + ' END'

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        return self._render_json_extract_from_binary(binary, operator, **kw)

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        return self._render_json_extract_from_binary(binary, operator, **kw)

    def visit_on_duplicate_key_update(self, on_duplicate, **kw):
        statement = self.current_executable
        if on_duplicate._parameter_ordering:
            parameter_ordering = [coercions.expect(roles.DMLColumnRole, key) for key in on_duplicate._parameter_ordering]
            ordered_keys = set(parameter_ordering)
            cols = [statement.table.c[key] for key in parameter_ordering if key in statement.table.c] + [c for c in statement.table.c if c.key not in ordered_keys]
        else:
            cols = statement.table.c
        clauses = []
        requires_mysql8_alias = self.dialect._requires_alias_for_on_duplicate_key
        if requires_mysql8_alias:
            if statement.table.name.lower() == 'new':
                _on_dup_alias_name = 'new_1'
            else:
                _on_dup_alias_name = 'new'
        for column in (col for col in cols if col.key in on_duplicate.update):
            val = on_duplicate.update[column.key]
            if coercions._is_literal(val):
                val = elements.BindParameter(None, val, type_=column.type)
                value_text = self.process(val.self_group(), use_schema=False)
            else:

                def replace(obj):
                    if isinstance(obj, elements.BindParameter) and obj.type._isnull:
                        obj = obj._clone()
                        obj.type = column.type
                        return obj
                    elif isinstance(obj, elements.ColumnClause) and obj.table is on_duplicate.inserted_alias:
                        if requires_mysql8_alias:
                            column_literal_clause = f'{_on_dup_alias_name}.{self.preparer.quote(obj.name)}'
                        else:
                            column_literal_clause = f'VALUES({self.preparer.quote(obj.name)})'
                        return literal_column(column_literal_clause)
                    else:
                        return None
                val = visitors.replacement_traverse(val, {}, replace)
                value_text = self.process(val.self_group(), use_schema=False)
            name_text = self.preparer.quote(column.name)
            clauses.append('%s = %s' % (name_text, value_text))
        non_matching = set(on_duplicate.update) - {c.key for c in cols}
        if non_matching:
            util.warn("Additional column names not matching any column keys in table '%s': %s" % (self.statement.table.name, ', '.join(("'%s'" % c for c in non_matching))))
        if requires_mysql8_alias:
            return f'AS {_on_dup_alias_name} ON DUPLICATE KEY UPDATE {', '.join(clauses)}'
        else:
            return f'ON DUPLICATE KEY UPDATE {', '.join(clauses)}'

    def visit_concat_op_expression_clauselist(self, clauselist, operator, **kw):
        return 'concat(%s)' % ', '.join((self.process(elem, **kw) for elem in clauselist.clauses))

    def visit_concat_op_binary(self, binary, operator, **kw):
        return 'concat(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
    _match_valid_flag_combinations = frozenset(((False, False, False), (True, False, False), (False, True, False), (False, False, True), (False, True, True)))
    _match_flag_expressions = ('IN BOOLEAN MODE', 'IN NATURAL LANGUAGE MODE', 'WITH QUERY EXPANSION')

    def visit_mysql_match(self, element, **kw):
        return self.visit_match_op_binary(element, element.operator, **kw)

    def visit_match_op_binary(self, binary, operator, **kw):
        """
        Note that `mysql_boolean_mode` is enabled by default because of
        backward compatibility
        """
        modifiers = binary.modifiers
        boolean_mode = modifiers.get('mysql_boolean_mode', True)
        natural_language = modifiers.get('mysql_natural_language', False)
        query_expansion = modifiers.get('mysql_query_expansion', False)
        flag_combination = (boolean_mode, natural_language, query_expansion)
        if flag_combination not in self._match_valid_flag_combinations:
            flags = ('in_boolean_mode=%s' % boolean_mode, 'in_natural_language_mode=%s' % natural_language, 'with_query_expansion=%s' % query_expansion)
            flags = ', '.join(flags)
            raise exc.CompileError('Invalid MySQL match flags: %s' % flags)
        match_clause = binary.left
        match_clause = self.process(match_clause, **kw)
        against_clause = self.process(binary.right, **kw)
        if any(flag_combination):
            flag_expressions = compress(self._match_flag_expressions, flag_combination)
            against_clause = [against_clause]
            against_clause.extend(flag_expressions)
            against_clause = ' '.join(against_clause)
        return 'MATCH (%s) AGAINST (%s)' % (match_clause, against_clause)

    def get_from_hint_text(self, table, text):
        return text

    def visit_typeclause(self, typeclause, type_=None, **kw):
        if type_ is None:
            type_ = typeclause.type.dialect_impl(self.dialect)
        if isinstance(type_, sqltypes.TypeDecorator):
            return self.visit_typeclause(typeclause, type_.impl, **kw)
        elif isinstance(type_, sqltypes.Integer):
            if getattr(type_, 'unsigned', False):
                return 'UNSIGNED INTEGER'
            else:
                return 'SIGNED INTEGER'
        elif isinstance(type_, sqltypes.TIMESTAMP):
            return 'DATETIME'
        elif isinstance(type_, (sqltypes.DECIMAL, sqltypes.DateTime, sqltypes.Date, sqltypes.Time)):
            return self.dialect.type_compiler_instance.process(type_)
        elif isinstance(type_, sqltypes.String) and (not isinstance(type_, (ENUM, SET))):
            adapted = CHAR._adapt_string_for_cast(type_)
            return self.dialect.type_compiler_instance.process(adapted)
        elif isinstance(type_, sqltypes._Binary):
            return 'BINARY'
        elif isinstance(type_, sqltypes.JSON):
            return 'JSON'
        elif isinstance(type_, sqltypes.NUMERIC):
            return self.dialect.type_compiler_instance.process(type_).replace('NUMERIC', 'DECIMAL')
        elif isinstance(type_, sqltypes.Float) and self.dialect._support_float_cast:
            return self.dialect.type_compiler_instance.process(type_)
        else:
            return None

    def visit_cast(self, cast, **kw):
        type_ = self.process(cast.typeclause)
        if type_ is None:
            util.warn('Datatype %s does not support CAST on MySQL/MariaDb; the CAST will be skipped.' % self.dialect.type_compiler_instance.process(cast.typeclause.type))
            return self.process(cast.clause.self_group(), **kw)
        return 'CAST(%s AS %s)' % (self.process(cast.clause, **kw), type_)

    def render_literal_value(self, value, type_):
        value = super().render_literal_value(value, type_)
        if self.dialect._backslash_escapes:
            value = value.replace('\\', '\\\\')
        return value

    def visit_true(self, element, **kw):
        return 'true'

    def visit_false(self, element, **kw):
        return 'false'

    def get_select_precolumns(self, select, **kw):
        """Add special MySQL keywords in place of DISTINCT.

        .. deprecated:: 1.4  This usage is deprecated.
           :meth:`_expression.Select.prefix_with` should be used for special
           keywords at the start of a SELECT.

        """
        if isinstance(select._distinct, str):
            util.warn_deprecated("Sending string values for 'distinct' is deprecated in the MySQL dialect and will be removed in a future release.  Please use :meth:`.Select.prefix_with` for special keywords at the start of a SELECT statement", version='1.4')
            return select._distinct.upper() + ' '
        return super().get_select_precolumns(select, **kw)

    def visit_join(self, join, asfrom=False, from_linter=None, **kwargs):
        if from_linter:
            from_linter.edges.add((join.left, join.right))
        if join.full:
            join_type = ' FULL OUTER JOIN '
        elif join.isouter:
            join_type = ' LEFT OUTER JOIN '
        else:
            join_type = ' INNER JOIN '
        return ''.join((self.process(join.left, asfrom=True, from_linter=from_linter, **kwargs), join_type, self.process(join.right, asfrom=True, from_linter=from_linter, **kwargs), ' ON ', self.process(join.onclause, from_linter=from_linter, **kwargs)))

    def for_update_clause(self, select, **kw):
        if select._for_update_arg.read:
            tmp = ' LOCK IN SHARE MODE'
        else:
            tmp = ' FOR UPDATE'
        if select._for_update_arg.of and self.dialect.supports_for_update_of:
            tables = util.OrderedSet()
            for c in select._for_update_arg.of:
                tables.update(sql_util.surface_selectables_only(c))
            tmp += ' OF ' + ', '.join((self.process(table, ashint=True, use_schema=False, **kw) for table in tables))
        if select._for_update_arg.nowait:
            tmp += ' NOWAIT'
        if select._for_update_arg.skip_locked:
            tmp += ' SKIP LOCKED'
        return tmp

    def limit_clause(self, select, **kw):
        limit_clause, offset_clause = (select._limit_clause, select._offset_clause)
        if limit_clause is None and offset_clause is None:
            return ''
        elif offset_clause is not None:
            if limit_clause is None:
                return ' \n LIMIT %s, %s' % (self.process(offset_clause, **kw), '18446744073709551615')
            else:
                return ' \n LIMIT %s, %s' % (self.process(offset_clause, **kw), self.process(limit_clause, **kw))
        else:
            return ' \n LIMIT %s' % (self.process(limit_clause, **kw),)

    def update_limit_clause(self, update_stmt):
        limit = update_stmt.kwargs.get('%s_limit' % self.dialect.name, None)
        if limit:
            return 'LIMIT %s' % limit
        else:
            return None

    def update_tables_clause(self, update_stmt, from_table, extra_froms, **kw):
        kw['asfrom'] = True
        return ', '.join((t._compiler_dispatch(self, **kw) for t in [from_table] + list(extra_froms)))

    def update_from_clause(self, update_stmt, from_table, extra_froms, from_hints, **kw):
        return None

    def delete_table_clause(self, delete_stmt, from_table, extra_froms, **kw):
        """If we have extra froms make sure we render any alias as hint."""
        ashint = False
        if extra_froms:
            ashint = True
        return from_table._compiler_dispatch(self, asfrom=True, iscrud=True, ashint=ashint, **kw)

    def delete_extra_from_clause(self, delete_stmt, from_table, extra_froms, from_hints, **kw):
        """Render the DELETE .. USING clause specific to MySQL."""
        kw['asfrom'] = True
        return 'USING ' + ', '.join((t._compiler_dispatch(self, fromhints=from_hints, **kw) for t in [from_table] + extra_froms))

    def visit_empty_set_expr(self, element_types, **kw):
        return 'SELECT %(outer)s FROM (SELECT %(inner)s) as _empty_set WHERE 1!=1' % {'inner': ', '.join(('1 AS _in_%s' % idx for idx, type_ in enumerate(element_types))), 'outer': ', '.join(('_in_%s' % idx for idx, type_ in enumerate(element_types)))}

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return 'NOT (%s <=> %s)' % (self.process(binary.left), self.process(binary.right))

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return '%s <=> %s' % (self.process(binary.left), self.process(binary.right))

    def _mariadb_regexp_flags(self, flags, pattern, **kw):
        return "CONCAT('(?', %s, ')', %s)" % (self.render_literal_value(flags, sqltypes.STRINGTYPE), self.process(pattern, **kw))

    def _regexp_match(self, op_string, binary, operator, **kw):
        flags = binary.modifiers['flags']
        if flags is None:
            return self._generate_generic_binary(binary, op_string, **kw)
        elif self.dialect.is_mariadb:
            return '%s%s%s' % (self.process(binary.left, **kw), op_string, self._mariadb_regexp_flags(flags, binary.right))
        else:
            text = 'REGEXP_LIKE(%s, %s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw), self.render_literal_value(flags, sqltypes.STRINGTYPE))
            if op_string == ' NOT REGEXP ':
                return 'NOT %s' % text
            else:
                return text

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        return self._regexp_match(' REGEXP ', binary, operator, **kw)

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return self._regexp_match(' NOT REGEXP ', binary, operator, **kw)

    def visit_regexp_replace_op_binary(self, binary, operator, **kw):
        flags = binary.modifiers['flags']
        if flags is None:
            return 'REGEXP_REPLACE(%s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw))
        elif self.dialect.is_mariadb:
            return 'REGEXP_REPLACE(%s, %s, %s)' % (self.process(binary.left, **kw), self._mariadb_regexp_flags(flags, binary.right.clauses[0]), self.process(binary.right.clauses[1], **kw))
        else:
            return 'REGEXP_REPLACE(%s, %s, %s)' % (self.process(binary.left, **kw), self.process(binary.right, **kw), self.render_literal_value(flags, sqltypes.STRINGTYPE))