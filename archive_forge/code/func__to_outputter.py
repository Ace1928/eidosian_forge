import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions._utils import (
from fugue.extensions.outputter.outputter import Outputter
def _to_outputter(obj: Any, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None, validation_rules: Optional[Dict[str, Any]]=None) -> Outputter:
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    load_namespace_extensions(obj)
    obj = parse_outputter(obj)
    exp: Optional[Exception] = None
    if validation_rules is None:
        validation_rules = {}
    try:
        return copy.copy(to_instance(obj, Outputter, global_vars=global_vars, local_vars=local_vars))
    except Exception as e:
        exp = e
    try:
        f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
        if isinstance(f, Outputter):
            return copy.copy(f)
        return _FuncAsOutputter.from_func(f, validation_rules=validation_rules)
    except Exception as e:
        exp = e
    raise FugueInterfacelessError(f'{obj} is not a valid outputter', exp)