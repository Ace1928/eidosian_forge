import inspect
from typing import (
import pandas as pd
import pyarrow as pa
from triad import Schema, assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.iter import EmptyAwareIterable, make_empty_aware
from ..constants import FUGUE_ENTRYPOINT
from ..dataset.api import count as df_count
from .array_dataframe import ArrayDataFrame
from .arrow_dataframe import ArrowDataFrame
from .dataframe import AnyDataFrame, DataFrame, LocalDataFrame, as_fugue_df
from .dataframe_iterable_dataframe import (
from .dataframes import DataFrames
from .iterable_dataframe import IterableDataFrame
from .pandas_dataframe import PandasDataFrame
@function_wrapper(FUGUE_ENTRYPOINT)
class DataFrameFunctionWrapper(FunctionWrapper):

    @property
    def need_output_schema(self) -> Optional[bool]:
        return self._rt.need_schema() if isinstance(self._rt, _DataFrameParamBase) else False

    def get_format_hint(self) -> Optional[str]:
        for v in self._params.values():
            if isinstance(v, _DataFrameParamBase):
                if v.format_hint() is not None:
                    return v.format_hint()
        if isinstance(self._rt, _DataFrameParamBase):
            return self._rt.format_hint()
        return None

    def run(self, args: List[Any], kwargs: Dict[str, Any], ignore_unknown: bool=False, output_schema: Any=None, output: bool=True, ctx: Any=None) -> Any:
        p: Dict[str, Any] = {}
        for i in range(len(args)):
            p[self._params.get_key_by_index(i)] = args[i]
        p.update(kwargs)
        has_kw = False
        rargs: Dict[str, Any] = {}
        for k, v in self._params.items():
            if isinstance(v, (PositionalParam, KeywordParam)):
                if isinstance(v, KeywordParam):
                    has_kw = True
            elif k in p:
                if isinstance(v, _DataFrameParamBase):
                    assert_or_throw(isinstance(p[k], DataFrame), lambda: TypeError(f'{p[k]} is not a DataFrame'))
                    rargs[k] = v.to_input_data(p[k], ctx=ctx)
                else:
                    rargs[k] = p[k]
                del p[k]
            elif v.required:
                raise ValueError(f'{k} is required by not given')
        if has_kw:
            rargs.update(p)
        elif not ignore_unknown and len(p) > 0:
            raise ValueError(f'{p} are not acceptable parameters')
        rt = self._func(**rargs)
        if not output:
            if isinstance(self._rt, _DataFrameParamBase):
                self._rt.count(rt)
            return
        if isinstance(self._rt, _DataFrameParamBase):
            return self._rt.to_output_df(rt, output_schema, ctx=ctx)
        return rt