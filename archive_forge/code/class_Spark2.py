from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
class Spark2(Hive):

    class Parser(Hive.Parser):
        TRIM_PATTERN_FIRST = True
        FUNCTIONS = {**Hive.Parser.FUNCTIONS, 'AGGREGATE': exp.Reduce.from_arg_list, 'APPROX_PERCENTILE': exp.ApproxQuantile.from_arg_list, 'BOOLEAN': _build_as_cast('boolean'), 'DATE': _build_as_cast('date'), 'DATE_TRUNC': lambda args: exp.TimestampTrunc(this=seq_get(args, 1), unit=exp.var(seq_get(args, 0))), 'DAYOFMONTH': lambda args: exp.DayOfMonth(this=exp.TsOrDsToDate(this=seq_get(args, 0))), 'DAYOFWEEK': lambda args: exp.DayOfWeek(this=exp.TsOrDsToDate(this=seq_get(args, 0))), 'DAYOFYEAR': lambda args: exp.DayOfYear(this=exp.TsOrDsToDate(this=seq_get(args, 0))), 'DOUBLE': _build_as_cast('double'), 'FLOAT': _build_as_cast('float'), 'FROM_UTC_TIMESTAMP': lambda args: exp.AtTimeZone(this=exp.cast(seq_get(args, 0) or exp.Var(this=''), exp.DataType.Type.TIMESTAMP), zone=seq_get(args, 1)), 'INT': _build_as_cast('int'), 'MAP_FROM_ARRAYS': exp.Map.from_arg_list, 'RLIKE': exp.RegexpLike.from_arg_list, 'SHIFTLEFT': binary_from_function(exp.BitwiseLeftShift), 'SHIFTRIGHT': binary_from_function(exp.BitwiseRightShift), 'STRING': _build_as_cast('string'), 'TIMESTAMP': _build_as_cast('timestamp'), 'TO_TIMESTAMP': lambda args: _build_as_cast('timestamp')(args) if len(args) == 1 else build_formatted_time(exp.StrToTime, 'spark')(args), 'TO_UNIX_TIMESTAMP': exp.StrToUnix.from_arg_list, 'TO_UTC_TIMESTAMP': lambda args: exp.FromTimeZone(this=exp.cast(seq_get(args, 0) or exp.Var(this=''), exp.DataType.Type.TIMESTAMP), zone=seq_get(args, 1)), 'TRUNC': lambda args: exp.DateTrunc(unit=seq_get(args, 1), this=seq_get(args, 0)), 'WEEKOFYEAR': lambda args: exp.WeekOfYear(this=exp.TsOrDsToDate(this=seq_get(args, 0)))}
        FUNCTION_PARSERS = {**Hive.Parser.FUNCTION_PARSERS, 'BROADCAST': lambda self: self._parse_join_hint('BROADCAST'), 'BROADCASTJOIN': lambda self: self._parse_join_hint('BROADCASTJOIN'), 'MAPJOIN': lambda self: self._parse_join_hint('MAPJOIN'), 'MERGE': lambda self: self._parse_join_hint('MERGE'), 'SHUFFLEMERGE': lambda self: self._parse_join_hint('SHUFFLEMERGE'), 'MERGEJOIN': lambda self: self._parse_join_hint('MERGEJOIN'), 'SHUFFLE_HASH': lambda self: self._parse_join_hint('SHUFFLE_HASH'), 'SHUFFLE_REPLICATE_NL': lambda self: self._parse_join_hint('SHUFFLE_REPLICATE_NL')}

        def _parse_add_column(self) -> t.Optional[exp.Expression]:
            return self._match_text_seq('ADD', 'COLUMNS') and self._parse_schema()

        def _parse_drop_column(self) -> t.Optional[exp.Drop | exp.Command]:
            return self._match_text_seq('DROP', 'COLUMNS') and self.expression(exp.Drop, this=self._parse_schema(), kind='COLUMNS')

        def _pivot_column_names(self, aggregations: t.List[exp.Expression]) -> t.List[str]:
            if len(aggregations) == 1:
                return ['']
            return pivot_column_names(aggregations, dialect='spark')

    class Generator(Hive.Generator):
        QUERY_HINTS = True
        NVL2_SUPPORTED = True
        CAN_IMPLEMENT_ARRAY_ANY = True
        PROPERTIES_LOCATION = {**Hive.Generator.PROPERTIES_LOCATION, exp.EngineProperty: exp.Properties.Location.UNSUPPORTED, exp.AutoIncrementProperty: exp.Properties.Location.UNSUPPORTED, exp.CharacterSetProperty: exp.Properties.Location.UNSUPPORTED, exp.CollateProperty: exp.Properties.Location.UNSUPPORTED}
        TRANSFORMS = {**Hive.Generator.TRANSFORMS, exp.ApproxDistinct: rename_func('APPROX_COUNT_DISTINCT'), exp.ArraySum: lambda self, e: f'AGGREGATE({self.sql(e, 'this')}, 0, (acc, x) -> acc + x, acc -> acc)', exp.ArrayToString: rename_func('ARRAY_JOIN'), exp.AtTimeZone: lambda self, e: self.func('FROM_UTC_TIMESTAMP', e.this, e.args.get('zone')), exp.BitwiseLeftShift: rename_func('SHIFTLEFT'), exp.BitwiseRightShift: rename_func('SHIFTRIGHT'), exp.Create: preprocess([remove_unique_constraints, lambda e: ctas_with_tmp_tables_to_create_tmp_view(e, temporary_storage_provider), move_schema_columns_to_partitioned_by]), exp.DateFromParts: rename_func('MAKE_DATE'), exp.DateTrunc: lambda self, e: self.func('TRUNC', e.this, unit_to_str(e)), exp.DayOfMonth: rename_func('DAYOFMONTH'), exp.DayOfWeek: rename_func('DAYOFWEEK'), exp.DayOfYear: rename_func('DAYOFYEAR'), exp.FileFormatProperty: lambda self, e: f'USING {e.name.upper()}', exp.From: transforms.preprocess([_unalias_pivot]), exp.FromTimeZone: lambda self, e: self.func('TO_UTC_TIMESTAMP', e.this, e.args.get('zone')), exp.LogicalAnd: rename_func('BOOL_AND'), exp.LogicalOr: rename_func('BOOL_OR'), exp.Map: _map_sql, exp.Pivot: transforms.preprocess([_unqualify_pivot_columns]), exp.Reduce: rename_func('AGGREGATE'), exp.RegexpReplace: lambda self, e: self.func('REGEXP_REPLACE', e.this, e.expression, e.args['replacement'], e.args.get('position')), exp.StrToDate: _str_to_date, exp.StrToTime: lambda self, e: self.func('TO_TIMESTAMP', e.this, self.format_time(e)), exp.TimestampTrunc: lambda self, e: self.func('DATE_TRUNC', unit_to_str(e), e.this), exp.Trim: trim_sql, exp.UnixToTime: _unix_to_time_sql, exp.VariancePop: rename_func('VAR_POP'), exp.WeekOfYear: rename_func('WEEKOFYEAR'), exp.WithinGroup: transforms.preprocess([transforms.remove_within_group_for_percentiles])}
        TRANSFORMS.pop(exp.ArraySort)
        TRANSFORMS.pop(exp.ILike)
        TRANSFORMS.pop(exp.Left)
        TRANSFORMS.pop(exp.MonthsBetween)
        TRANSFORMS.pop(exp.Right)
        WRAP_DERIVED_VALUES = False
        CREATE_FUNCTION_RETURN_AS = False

        def struct_sql(self, expression: exp.Struct) -> str:
            from sqlglot.generator import Generator
            return Generator.struct_sql(self, expression)

        def cast_sql(self, expression: exp.Cast, safe_prefix: t.Optional[str]=None) -> str:
            if is_parse_json(expression.this):
                schema = f"'{self.sql(expression, 'to')}'"
                return self.func('FROM_JSON', expression.this.this, schema)
            if is_parse_json(expression):
                return self.func('TO_JSON', expression.this)
            return super(Hive.Generator, self).cast_sql(expression, safe_prefix=safe_prefix)

        def columndef_sql(self, expression: exp.ColumnDef, sep: str=' ') -> str:
            return super().columndef_sql(expression, sep=': ' if isinstance(expression.parent, exp.DataType) and expression.parent.is_type('struct') else sep)

    class Tokenizer(Hive.Tokenizer):
        HEX_STRINGS = [("X'", "'")]