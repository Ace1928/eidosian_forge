import copy
from typing import Any, Callable, Dict, List, Optional, Type, Union, no_type_check
from triad import ParamDict, Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import is_class_method, parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
from fugue.extensions.transformer.transformer import CoTransformer, Transformer
from .._utils import (
class _FuncAsCoTransformer(CoTransformer):

    def validate_on_compile(self) -> None:
        super().validate_on_compile()
        _validate_callback(self)

    def get_output_schema(self, dfs: DataFrames) -> Any:
        return self._parse_schema(self._output_schema_arg, dfs)

    def get_format_hint(self) -> Optional[str]:
        return self._format_hint

    @property
    def validation_rules(self) -> ParamDict:
        return self._validation_rules

    @no_type_check
    def transform(self, dfs: DataFrames) -> LocalDataFrame:
        cb = _get_callback(self)
        if self._dfs_input:
            return self._wrapper.run([dfs] + cb, self.params, ignore_unknown=False, output_schema=self.output_schema)
        if not dfs.has_key:
            return self._wrapper.run(list(dfs.values()) + cb, self.params, ignore_unknown=False, output_schema=self.output_schema)
        else:
            p = dict(dfs)
            p.update(self.params)
            return self._wrapper.run([] + cb, p, ignore_unknown=False, output_schema=self.output_schema)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper(*args, **kwargs)

    @no_type_check
    def __uuid__(self) -> str:
        return to_uuid(self._wrapper.__uuid__(), self._output_schema_arg, self._dfs_input)

    def _parse_schema(self, obj: Any, dfs: DataFrames) -> Schema:
        if callable(obj):
            return obj(dfs, **self.params)
        if isinstance(obj, str):
            return Schema(obj)
        if isinstance(obj, List):
            s = Schema()
            for x in obj:
                s += self._parse_schema(x, dfs)
            return s
        return Schema(obj)

    @staticmethod
    def from_func(func: Callable, schema: Any, validation_rules: Dict[str, Any]) -> '_FuncAsCoTransformer':
        assert_or_throw(len(validation_rules) == 0, NotImplementedError('CoTransformer does not support validation rules'))
        if schema is None:
            schema = parse_output_schema_from_comment(func)
        if isinstance(schema, Schema):
            schema = str(schema)
        if isinstance(schema, str):
            assert_or_throw('*' not in schema, FugueInterfacelessError("* can't be used on cotransformer output schema"))
        assert_arg_not_none(schema, 'schema')
        tr = _FuncAsCoTransformer()
        tr._wrapper = DataFrameFunctionWrapper(func, '^(c|[lspq]+)[fF]?x*z?$', '^[lspq]$')
        tr._dfs_input = tr._wrapper.input_code[0] == 'c'
        tr._output_schema_arg = schema
        tr._validation_rules = {}
        tr._uses_callback = 'f' in tr._wrapper.input_code.lower()
        tr._requires_callback = 'F' in tr._wrapper.input_code
        tr._format_hint = tr._wrapper.get_format_hint()
        return tr