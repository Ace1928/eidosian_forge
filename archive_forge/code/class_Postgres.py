from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
class Postgres(Dialect):
    INDEX_OFFSET = 1
    TYPED_DIVISION = True
    CONCAT_COALESCE = True
    NULL_ORDERING = 'nulls_are_large'
    TIME_FORMAT = "'YYYY-MM-DD HH24:MI:SS'"
    TIME_MAPPING = {'AM': '%p', 'PM': '%p', 'D': '%u', 'DD': '%d', 'DDD': '%j', 'FMDD': '%-d', 'FMDDD': '%-j', 'FMHH12': '%-I', 'FMHH24': '%-H', 'FMMI': '%-M', 'FMMM': '%-m', 'FMSS': '%-S', 'HH12': '%I', 'HH24': '%H', 'MI': '%M', 'MM': '%m', 'OF': '%z', 'SS': '%S', 'TMDay': '%A', 'TMDy': '%a', 'TMMon': '%b', 'TMMonth': '%B', 'TZ': '%Z', 'US': '%f', 'WW': '%U', 'YY': '%y', 'YYYY': '%Y'}

    class Tokenizer(tokens.Tokenizer):
        BIT_STRINGS = [("b'", "'"), ("B'", "'")]
        HEX_STRINGS = [("x'", "'"), ("X'", "'")]
        BYTE_STRINGS = [("e'", "'"), ("E'", "'")]
        HEREDOC_STRINGS = ['$']
        HEREDOC_TAG_IS_IDENTIFIER = True
        HEREDOC_STRING_ALTERNATIVE = TokenType.PARAMETER
        KEYWORDS = {**tokens.Tokenizer.KEYWORDS, '~~': TokenType.LIKE, '~~*': TokenType.ILIKE, '~*': TokenType.IRLIKE, '~': TokenType.RLIKE, '@@': TokenType.DAT, '@>': TokenType.AT_GT, '<@': TokenType.LT_AT, '|/': TokenType.PIPE_SLASH, '||/': TokenType.DPIPE_SLASH, 'BEGIN': TokenType.COMMAND, 'BEGIN TRANSACTION': TokenType.BEGIN, 'BIGSERIAL': TokenType.BIGSERIAL, 'CHARACTER VARYING': TokenType.VARCHAR, 'CONSTRAINT TRIGGER': TokenType.COMMAND, 'DECLARE': TokenType.COMMAND, 'DO': TokenType.COMMAND, 'EXEC': TokenType.COMMAND, 'HSTORE': TokenType.HSTORE, 'JSONB': TokenType.JSONB, 'MONEY': TokenType.MONEY, 'REFRESH': TokenType.COMMAND, 'REINDEX': TokenType.COMMAND, 'RESET': TokenType.COMMAND, 'REVOKE': TokenType.COMMAND, 'SERIAL': TokenType.SERIAL, 'SMALLSERIAL': TokenType.SMALLSERIAL, 'NAME': TokenType.NAME, 'TEMP': TokenType.TEMPORARY, 'CSTRING': TokenType.PSEUDO_TYPE, 'OID': TokenType.OBJECT_IDENTIFIER, 'ONLY': TokenType.ONLY, 'OPERATOR': TokenType.OPERATOR, 'REGCLASS': TokenType.OBJECT_IDENTIFIER, 'REGCOLLATION': TokenType.OBJECT_IDENTIFIER, 'REGCONFIG': TokenType.OBJECT_IDENTIFIER, 'REGDICTIONARY': TokenType.OBJECT_IDENTIFIER, 'REGNAMESPACE': TokenType.OBJECT_IDENTIFIER, 'REGOPER': TokenType.OBJECT_IDENTIFIER, 'REGOPERATOR': TokenType.OBJECT_IDENTIFIER, 'REGPROC': TokenType.OBJECT_IDENTIFIER, 'REGPROCEDURE': TokenType.OBJECT_IDENTIFIER, 'REGROLE': TokenType.OBJECT_IDENTIFIER, 'REGTYPE': TokenType.OBJECT_IDENTIFIER}
        SINGLE_TOKENS = {**tokens.Tokenizer.SINGLE_TOKENS, '$': TokenType.HEREDOC_STRING}
        VAR_SINGLE_TOKENS = {'$'}

    class Parser(parser.Parser):
        PROPERTY_PARSERS = {**parser.Parser.PROPERTY_PARSERS, 'SET': lambda self: self.expression(exp.SetConfigProperty, this=self._parse_set())}
        PROPERTY_PARSERS.pop('INPUT')
        FUNCTIONS = {**parser.Parser.FUNCTIONS, 'DATE_TRUNC': build_timestamp_trunc, 'GENERATE_SERIES': _build_generate_series, 'JSON_EXTRACT_PATH': build_json_extract_path(exp.JSONExtract), 'JSON_EXTRACT_PATH_TEXT': build_json_extract_path(exp.JSONExtractScalar), 'MAKE_TIME': exp.TimeFromParts.from_arg_list, 'MAKE_TIMESTAMP': exp.TimestampFromParts.from_arg_list, 'NOW': exp.CurrentTimestamp.from_arg_list, 'TO_CHAR': build_formatted_time(exp.TimeToStr, 'postgres'), 'TO_TIMESTAMP': _build_to_timestamp, 'UNNEST': exp.Explode.from_arg_list}
        FUNCTION_PARSERS = {**parser.Parser.FUNCTION_PARSERS, 'DATE_PART': lambda self: self._parse_date_part()}
        BITWISE = {**parser.Parser.BITWISE, TokenType.HASH: exp.BitwiseXor}
        EXPONENT = {TokenType.CARET: exp.Pow}
        RANGE_PARSERS = {**parser.Parser.RANGE_PARSERS, TokenType.AT_GT: binary_range_parser(exp.ArrayContains), TokenType.DAMP: binary_range_parser(exp.ArrayOverlaps), TokenType.DAT: lambda self, this: self.expression(exp.MatchAgainst, this=self._parse_bitwise(), expressions=[this]), TokenType.LT_AT: binary_range_parser(exp.ArrayContained), TokenType.OPERATOR: lambda self, this: self._parse_operator(this)}
        STATEMENT_PARSERS = {**parser.Parser.STATEMENT_PARSERS, TokenType.END: lambda self: self._parse_commit_or_rollback()}
        JSON_ARROWS_REQUIRE_JSON_TYPE = True
        COLUMN_OPERATORS = {**parser.Parser.COLUMN_OPERATORS, TokenType.ARROW: lambda self, this, path: build_json_extract_path(exp.JSONExtract, arrow_req_json_type=self.JSON_ARROWS_REQUIRE_JSON_TYPE)([this, path]), TokenType.DARROW: lambda self, this, path: build_json_extract_path(exp.JSONExtractScalar, arrow_req_json_type=self.JSON_ARROWS_REQUIRE_JSON_TYPE)([this, path])}

        def _parse_operator(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
            while True:
                if not self._match(TokenType.L_PAREN):
                    break
                op = ''
                while self._curr and (not self._match(TokenType.R_PAREN)):
                    op += self._curr.text
                    self._advance()
                this = self.expression(exp.Operator, comments=self._prev_comments, this=this, operator=op, expression=self._parse_bitwise())
                if not self._match(TokenType.OPERATOR):
                    break
            return this

        def _parse_date_part(self) -> exp.Expression:
            part = self._parse_type()
            self._match(TokenType.COMMA)
            value = self._parse_bitwise()
            if part and part.is_string:
                part = exp.var(part.name)
            return self.expression(exp.Extract, this=part, expression=value)

    class Generator(generator.Generator):
        SINGLE_STRING_INTERVAL = True
        RENAME_TABLE_WITH_DB = False
        LOCKING_READS_SUPPORTED = True
        JOIN_HINTS = False
        TABLE_HINTS = False
        QUERY_HINTS = False
        NVL2_SUPPORTED = False
        PARAMETER_TOKEN = '$'
        TABLESAMPLE_SIZE_IS_ROWS = False
        TABLESAMPLE_SEED_KEYWORD = 'REPEATABLE'
        SUPPORTS_SELECT_INTO = True
        JSON_TYPE_REQUIRED_FOR_EXTRACTION = True
        SUPPORTS_UNLOGGED_TABLES = True
        LIKE_PROPERTY_INSIDE_SCHEMA = True
        MULTI_ARG_DISTINCT = False
        CAN_IMPLEMENT_ARRAY_ANY = True
        SUPPORTED_JSON_PATH_PARTS = {exp.JSONPathKey, exp.JSONPathRoot, exp.JSONPathSubscript}
        TYPE_MAPPING = {**generator.Generator.TYPE_MAPPING, exp.DataType.Type.TINYINT: 'SMALLINT', exp.DataType.Type.FLOAT: 'REAL', exp.DataType.Type.DOUBLE: 'DOUBLE PRECISION', exp.DataType.Type.BINARY: 'BYTEA', exp.DataType.Type.VARBINARY: 'BYTEA', exp.DataType.Type.DATETIME: 'TIMESTAMP'}
        TRANSFORMS = {**generator.Generator.TRANSFORMS, exp.AnyValue: any_value_to_max_sql, exp.Array: lambda self, e: f'{self.normalize_func('ARRAY')}({self.sql(e.expressions[0])})' if isinstance(seq_get(e.expressions, 0), exp.Select) else f'{self.normalize_func('ARRAY')}[{self.expressions(e, flat=True)}]', exp.ArrayConcat: rename_func('ARRAY_CAT'), exp.ArrayContained: lambda self, e: self.binary(e, '<@'), exp.ArrayContains: lambda self, e: self.binary(e, '@>'), exp.ArrayOverlaps: lambda self, e: self.binary(e, '&&'), exp.ArrayFilter: filter_array_using_unnest, exp.ArraySize: lambda self, e: self.func('ARRAY_LENGTH', e.this, e.expression or '1'), exp.BitwiseXor: lambda self, e: self.binary(e, '#'), exp.ColumnDef: transforms.preprocess([_auto_increment_to_serial, _serial_to_generated]), exp.CurrentDate: no_paren_current_date_sql, exp.CurrentTimestamp: lambda *_: 'CURRENT_TIMESTAMP', exp.CurrentUser: lambda *_: 'CURRENT_USER', exp.DateAdd: _date_add_sql('+'), exp.DateDiff: _date_diff_sql, exp.DateStrToDate: datestrtodate_sql, exp.DataType: _datatype_sql, exp.DateSub: _date_add_sql('-'), exp.Explode: rename_func('UNNEST'), exp.GroupConcat: _string_agg_sql, exp.JSONExtract: _json_extract_sql('JSON_EXTRACT_PATH', '->'), exp.JSONExtractScalar: _json_extract_sql('JSON_EXTRACT_PATH_TEXT', '->>'), exp.JSONBExtract: lambda self, e: self.binary(e, '#>'), exp.JSONBExtractScalar: lambda self, e: self.binary(e, '#>>'), exp.JSONBContains: lambda self, e: self.binary(e, '?'), exp.ParseJSON: lambda self, e: self.sql(exp.cast(e.this, exp.DataType.Type.JSON)), exp.JSONPathKey: json_path_key_only_name, exp.JSONPathRoot: lambda *_: '', exp.JSONPathSubscript: lambda self, e: self.json_path_part(e.this), exp.LastDay: no_last_day_sql, exp.LogicalOr: rename_func('BOOL_OR'), exp.LogicalAnd: rename_func('BOOL_AND'), exp.Max: max_or_greatest, exp.MapFromEntries: no_map_from_entries_sql, exp.Min: min_or_least, exp.Merge: merge_without_target_sql, exp.PartitionedByProperty: lambda self, e: f'PARTITION BY {self.sql(e, 'this')}', exp.PercentileCont: transforms.preprocess([transforms.add_within_group_for_percentiles]), exp.PercentileDisc: transforms.preprocess([transforms.add_within_group_for_percentiles]), exp.Pivot: no_pivot_sql, exp.Pow: lambda self, e: self.binary(e, '^'), exp.Rand: rename_func('RANDOM'), exp.RegexpLike: lambda self, e: self.binary(e, '~'), exp.RegexpILike: lambda self, e: self.binary(e, '~*'), exp.Select: transforms.preprocess([transforms.eliminate_semi_and_anti_joins, transforms.eliminate_qualify]), exp.StrPosition: str_position_sql, exp.StrToDate: lambda self, e: self.func('TO_DATE', e.this, self.format_time(e)), exp.StrToTime: lambda self, e: self.func('TO_TIMESTAMP', e.this, self.format_time(e)), exp.StructExtract: struct_extract_sql, exp.Substring: _substring_sql, exp.TimeFromParts: rename_func('MAKE_TIME'), exp.TimestampFromParts: rename_func('MAKE_TIMESTAMP'), exp.TimestampTrunc: timestamptrunc_sql, exp.TimeStrToTime: timestrtotime_sql, exp.TimeToStr: lambda self, e: self.func('TO_CHAR', e.this, self.format_time(e)), exp.ToChar: lambda self, e: self.function_fallback_sql(e), exp.Trim: trim_sql, exp.TryCast: no_trycast_sql, exp.TsOrDsAdd: _date_add_sql('+'), exp.TsOrDsDiff: _date_diff_sql, exp.UnixToTime: lambda self, e: self.func('TO_TIMESTAMP', e.this), exp.TimeToUnix: lambda self, e: self.func('DATE_PART', exp.Literal.string('epoch'), e.this), exp.VariancePop: rename_func('VAR_POP'), exp.Variance: rename_func('VAR_SAMP'), exp.Xor: bool_xor_sql}
        PROPERTIES_LOCATION = {**generator.Generator.PROPERTIES_LOCATION, exp.PartitionedByProperty: exp.Properties.Location.POST_SCHEMA, exp.TransientProperty: exp.Properties.Location.UNSUPPORTED, exp.VolatileProperty: exp.Properties.Location.UNSUPPORTED}

        def unnest_sql(self, expression: exp.Unnest) -> str:
            if len(expression.expressions) == 1:
                from sqlglot.optimizer.annotate_types import annotate_types
                this = annotate_types(expression.expressions[0])
                if this.is_type('array<json>'):
                    while isinstance(this, exp.Cast):
                        this = this.this
                    arg = self.sql(exp.cast(this, exp.DataType.Type.JSON))
                    alias = self.sql(expression, 'alias')
                    alias = f' AS {alias}' if alias else ''
                    if expression.args.get('offset'):
                        self.unsupported('Unsupported JSON_ARRAY_ELEMENTS with offset')
                    return f'JSON_ARRAY_ELEMENTS({arg}){alias}'
            return super().unnest_sql(expression)

        def bracket_sql(self, expression: exp.Bracket) -> str:
            """Forms like ARRAY[1, 2, 3][3] aren't allowed; we need to wrap the ARRAY."""
            if isinstance(expression.this, exp.Array):
                expression.set('this', exp.paren(expression.this, copy=False))
            return super().bracket_sql(expression)

        def matchagainst_sql(self, expression: exp.MatchAgainst) -> str:
            this = self.sql(expression, 'this')
            expressions = [f'{self.sql(e)} @@ {this}' for e in expression.expressions]
            sql = ' OR '.join(expressions)
            return f'({sql})' if len(expressions) > 1 else sql