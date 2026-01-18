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
def _validate_callback(ctx: Any) -> None:
    if ctx._requires_callback:
        assert_or_throw(ctx.has_callback, FugueInterfacelessError(f'Callback is required but not provided: {ctx}'))