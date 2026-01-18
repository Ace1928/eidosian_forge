from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
class SQLite(Dialect):
    NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_INSENSITIVE
    SUPPORTS_SEMI_ANTI_JOIN = False
    TYPED_DIVISION = True
    SAFE_DIVISION = True

    class Tokenizer(tokens.Tokenizer):
        IDENTIFIERS = ['"', ('[', ']'), '`']
        HEX_STRINGS = [("x'", "'"), ("X'", "'"), ('0x', ''), ('0X', '')]

    class Parser(parser.Parser):
        FUNCTIONS = {**parser.Parser.FUNCTIONS, 'EDITDIST3': exp.Levenshtein.from_arg_list, 'STRFTIME': _build_strftime}
        STRING_ALIASES = True

    class Generator(generator.Generator):
        JOIN_HINTS = False
        TABLE_HINTS = False
        QUERY_HINTS = False
        NVL2_SUPPORTED = False
        JSON_PATH_BRACKETED_KEY_SUPPORTED = False
        SUPPORTS_CREATE_TABLE_LIKE = False
        SUPPORTS_TABLE_ALIAS_COLUMNS = False
        SUPPORTS_TO_NUMBER = False
        SUPPORTED_JSON_PATH_PARTS = {exp.JSONPathKey, exp.JSONPathRoot, exp.JSONPathSubscript}
        TYPE_MAPPING = {**generator.Generator.TYPE_MAPPING, exp.DataType.Type.BOOLEAN: 'INTEGER', exp.DataType.Type.TINYINT: 'INTEGER', exp.DataType.Type.SMALLINT: 'INTEGER', exp.DataType.Type.INT: 'INTEGER', exp.DataType.Type.BIGINT: 'INTEGER', exp.DataType.Type.FLOAT: 'REAL', exp.DataType.Type.DOUBLE: 'REAL', exp.DataType.Type.DECIMAL: 'REAL', exp.DataType.Type.CHAR: 'TEXT', exp.DataType.Type.NCHAR: 'TEXT', exp.DataType.Type.VARCHAR: 'TEXT', exp.DataType.Type.NVARCHAR: 'TEXT', exp.DataType.Type.BINARY: 'BLOB', exp.DataType.Type.VARBINARY: 'BLOB'}
        TOKEN_MAPPING = {TokenType.AUTO_INCREMENT: 'AUTOINCREMENT'}
        TRANSFORMS = {**generator.Generator.TRANSFORMS, exp.AnyValue: any_value_to_max_sql, exp.Concat: concat_to_dpipe_sql, exp.CountIf: count_if_to_sum, exp.Create: transforms.preprocess([_transform_create]), exp.CurrentDate: lambda *_: 'CURRENT_DATE', exp.CurrentTime: lambda *_: 'CURRENT_TIME', exp.CurrentTimestamp: lambda *_: 'CURRENT_TIMESTAMP', exp.DateAdd: _date_add_sql, exp.DateStrToDate: lambda self, e: self.sql(e, 'this'), exp.If: rename_func('IIF'), exp.ILike: no_ilike_sql, exp.JSONExtract: _json_extract_sql, exp.JSONExtractScalar: arrow_json_extract_sql, exp.Levenshtein: rename_func('EDITDIST3'), exp.LogicalOr: rename_func('MAX'), exp.LogicalAnd: rename_func('MIN'), exp.Pivot: no_pivot_sql, exp.Rand: rename_func('RANDOM'), exp.Select: transforms.preprocess([transforms.eliminate_distinct_on, transforms.eliminate_qualify, transforms.eliminate_semi_and_anti_joins]), exp.TableSample: no_tablesample_sql, exp.TimeStrToTime: lambda self, e: self.sql(e, 'this'), exp.TimeToStr: lambda self, e: self.func('STRFTIME', e.args.get('format'), e.this), exp.TryCast: no_trycast_sql, exp.TsOrDsToTimestamp: lambda self, e: self.sql(e, 'this')}
        PROPERTIES_LOCATION = {prop: exp.Properties.Location.UNSUPPORTED for prop in generator.Generator.PROPERTIES_LOCATION}
        PROPERTIES_LOCATION[exp.LikeProperty] = exp.Properties.Location.POST_SCHEMA
        PROPERTIES_LOCATION[exp.TemporaryProperty] = exp.Properties.Location.POST_CREATE
        LIMIT_FETCH = 'LIMIT'

        def cast_sql(self, expression: exp.Cast, safe_prefix: t.Optional[str]=None) -> str:
            if expression.is_type('date'):
                return self.func('DATE', expression.this)
            return super().cast_sql(expression)

        def generateseries_sql(self, expression: exp.GenerateSeries) -> str:
            parent = expression.parent
            alias = parent and parent.args.get('alias')
            if isinstance(alias, exp.TableAlias) and alias.columns:
                column_alias = alias.columns[0]
                alias.set('columns', None)
                sql = self.sql(exp.select(exp.alias_('value', column_alias)).from_(expression).subquery())
            else:
                sql = super().generateseries_sql(expression)
            return sql

        def datediff_sql(self, expression: exp.DateDiff) -> str:
            unit = expression.args.get('unit')
            unit = unit.name.upper() if unit else 'DAY'
            sql = f'(JULIANDAY({self.sql(expression, 'this')}) - JULIANDAY({self.sql(expression, 'expression')}))'
            if unit == 'MONTH':
                sql = f'{sql} / 30.0'
            elif unit == 'YEAR':
                sql = f'{sql} / 365.0'
            elif unit == 'HOUR':
                sql = f'{sql} * 24.0'
            elif unit == 'MINUTE':
                sql = f'{sql} * 1440.0'
            elif unit == 'SECOND':
                sql = f'{sql} * 86400.0'
            elif unit == 'MILLISECOND':
                sql = f'{sql} * 86400000.0'
            elif unit == 'MICROSECOND':
                sql = f'{sql} * 86400000000.0'
            elif unit == 'NANOSECOND':
                sql = f'{sql} * 8640000000000.0'
            else:
                self.unsupported("DATEDIFF unsupported for '{unit}'.")
            return f'CAST({sql} AS INTEGER)'

        def groupconcat_sql(self, expression: exp.GroupConcat) -> str:
            this = expression.this
            distinct = expression.find(exp.Distinct)
            if distinct:
                this = distinct.expressions[0]
                distinct_sql = 'DISTINCT '
            else:
                distinct_sql = ''
            if isinstance(expression.this, exp.Order):
                self.unsupported("SQLite GROUP_CONCAT doesn't support ORDER BY.")
                if expression.this.this and (not distinct):
                    this = expression.this.this
            separator = expression.args.get('separator')
            return f'GROUP_CONCAT({distinct_sql}{self.format_args(this, separator)})'

        def least_sql(self, expression: exp.Least) -> str:
            if len(expression.expressions) > 1:
                return rename_func('MIN')(self, expression)
            return self.sql(expression, 'this')

        def transaction_sql(self, expression: exp.Transaction) -> str:
            this = expression.this
            this = f' {this}' if this else ''
            return f'BEGIN{this} TRANSACTION'