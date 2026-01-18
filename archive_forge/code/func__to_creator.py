import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.creator.creator import Creator
from .._utils import load_namespace_extensions
def _to_creator(obj: Any, schema: Any=None, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> Creator:
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    load_namespace_extensions(obj)
    obj = parse_creator(obj)
    exp: Optional[Exception] = None
    try:
        return copy.copy(to_instance(obj, Creator, global_vars=global_vars, local_vars=local_vars))
    except Exception as e:
        exp = e
    try:
        f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
        if isinstance(f, Creator):
            return copy.copy(f)
        return _FuncAsCreator.from_func(f, schema)
    except Exception as e:
        exp = e
    raise FugueInterfacelessError(f'{obj} is not a valid creator', exp)