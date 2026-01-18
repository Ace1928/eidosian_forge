import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
@functools.lru_cache(None)
def _get_overridable_functions() -> Tuple[Dict[Any, List[Callable]], Dict[Callable, str]]:
    overridable_funcs = collections.defaultdict(list)
    index = {}
    tested_namespaces = [('torch', torch, torch.__all__), ('torch.functional', torch.functional, torch.functional.__all__), ('torch.nn.functional', torch.nn.functional, dir(torch.nn.functional)), ('torch.nn.init', torch.nn.init, dir(torch.nn.init)), ('torch.Tensor', torch.Tensor, dir(torch.Tensor)), ('torch.linalg', torch.linalg, dir(torch.linalg)), ('torch.fft', torch.fft, dir(torch.fft)), ('torch.special', torch.special, dir(torch.special))]
    for namespace_str, namespace, ns_funcs in tested_namespaces:
        for func_name in ns_funcs:
            ignore = False
            if namespace is not torch.Tensor:
                if func_name.startswith('__'):
                    continue
                elif func_name.startswith('_'):
                    ignore = True
                elif func_name.endswith('_'):
                    ignore = True
                elif not func_name[0].islower():
                    ignore = True
                elif func_name == 'unique_dim':
                    continue
            else:
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    continue
                if func_name == '__weakref__':
                    continue
            func = getattr(namespace, func_name)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                continue
            if isinstance(func, types.ModuleType):
                continue
            if isinstance(func, __future__._Feature):
                continue
            if not callable(func) and hasattr(func, '__get__'):
                index[func.__get__] = f'{namespace_str}.{func_name}.__get__'
                index[func.__set__] = f'{namespace_str}.{func_name}.__set__'
                if ignore:
                    continue
                if func.__get__ in get_ignored_functions():
                    msg = '{}.{} is in the tuple returned by torch._overrides.get_ignored_functions but still has an explicit override'
                    assert func.__get__ not in get_testing_overrides(), msg.format(namespace, func.__name__)
                    continue
                else:
                    overridable_funcs[func].append(func.__get__)
                    continue
            if not callable(func):
                continue
            index[func] = f'{namespace_str}.{func_name}'
            if ignore:
                continue
            if func in get_ignored_functions():
                msg = '{}.{} is in the tuple returned by torch._overrides.get_ignored_functions but still has an explicit override'
                assert func not in get_testing_overrides(), msg.format(namespace, func.__name__)
                continue
            overridable_funcs[namespace].append(func)
    return (overridable_funcs, index)