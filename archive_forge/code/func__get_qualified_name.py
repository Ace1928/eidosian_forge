from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
def _get_qualified_name(func: Callable[..., Any]) -> str:
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    if isinstance(func, (types.MethodDescriptorType, types.WrapperDescriptorType)) and func is getattr(torch.Tensor, func.__name__, None):
        return f'torch.Tensor.{func.__name__}'
    name = func.__name__
    if name == '<lambda>':
        try:
            name = inspect.getsource(func).split('=')[0].strip()
        except Exception as e:
            raise RuntimeError('Unable to represent lambda') from e
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')
    if module == 'torch' and name == 'segment_reduce':
        name = '_' + name
    return f'{module}.{name}'