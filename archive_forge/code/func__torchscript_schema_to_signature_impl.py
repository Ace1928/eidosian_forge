import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
def _torchscript_schema_to_signature_impl(ts_schema: torch._C.FunctionSchema) -> inspect.Signature:
    from inspect import Parameter
    parameters: List[Parameter] = []
    for arg in ts_schema.arguments:
        arg_type = _torchscript_type_to_python_type(arg.type)
        default = arg.default_value if arg.has_default_value() else Parameter.empty
        name = arg.name if arg.name != 'self' else 'input'
        kind = Parameter.KEYWORD_ONLY if arg.kwarg_only else Parameter.POSITIONAL_OR_KEYWORD
        if name == 'from':
            assert kind == Parameter.POSITIONAL_OR_KEYWORD
            kind = Parameter.POSITIONAL_ONLY
            for idx, p in enumerate(parameters):
                assert p.kind == Parameter.POSITIONAL_OR_KEYWORD
                parameters[idx] = Parameter(name=p.name, kind=Parameter.POSITIONAL_ONLY, default=p.default, annotation=p.annotation)
        parameters.append(Parameter(name=name, kind=kind, default=default, annotation=arg_type))
    return_types = [_torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns]
    if len(return_types) == 0:
        return_type = None
    elif len(return_types) == 1:
        return_type = return_types[0]
    else:
        return_type = tuple(return_types)
    return inspect.Signature(parameters, return_annotation=return_type)