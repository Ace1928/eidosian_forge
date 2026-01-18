import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame, DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.processor.processor import Processor
from .._utils import (
class _FuncAsProcessor(Processor):

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return self._validation_rules

    @no_type_check
    def process(self, dfs: DataFrames) -> DataFrame:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if self._engine_param is not None:
            args.append(self._engine_param.to_input(self.execution_engine))
        if self._use_dfs:
            args.append(dfs)
        elif not dfs.has_key:
            args += dfs.values()
        else:
            kwargs.update(dfs)
        kwargs.update(self.params)
        return self._wrapper.run(args=args, kwargs=kwargs, output_schema=self.output_schema if self._need_output_schema else None, ctx=self.execution_engine)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper(*args, **kwargs)

    @no_type_check
    def __uuid__(self) -> str:
        return to_uuid(self._wrapper, self._engine_param, self._use_dfs, self._need_output_schema, str(self._output_schema))

    @no_type_check
    @staticmethod
    def from_func(func: Callable, schema: Any, validation_rules: Dict[str, Any]) -> '_FuncAsProcessor':
        if schema is None:
            schema = parse_output_schema_from_comment(func)
        validation_rules.update(parse_validation_rules_from_comment(func))
        tr = _FuncAsProcessor()
        tr._wrapper = DataFrameFunctionWrapper(func, '^e?(c|[dlspq]+)x*z?$', '^[dlspq]$')
        tr._engine_param = tr._wrapper._params.get_value_by_index(0) if tr._wrapper.input_code.startswith('e') else None
        tr._use_dfs = 'c' in tr._wrapper.input_code
        tr._need_output_schema = tr._wrapper.need_output_schema
        tr._validation_rules = validation_rules
        tr._output_schema = Schema(schema)
        if len(tr._output_schema) == 0:
            assert_or_throw(tr._need_output_schema is None or not tr._need_output_schema, FugueInterfacelessError(f'schema must be provided for return type {tr._wrapper._rt}'))
        else:
            assert_or_throw(tr._need_output_schema is None or tr._need_output_schema, FugueInterfacelessError(f'schema must not be provided for return type {tr._wrapper._rt}'))
        return tr