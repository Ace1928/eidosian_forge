import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _infer_injected_bindings(callable: Callable, only_explicit_bindings: bool) -> Dict[str, type]:

    def _is_new_union_type(instance: Any) -> bool:
        new_union_type = getattr(types, 'UnionType', None)
        return new_union_type is not None and isinstance(instance, new_union_type)
    spec = inspect.getfullargspec(callable)
    try:
        bindings = get_type_hints(cast(Callable, _NoReturnAnnotationProxy(callable)), include_extras=True)
    except NameError as e:
        raise _BindingNotYetAvailable(e)
    bindings.pop('return', None)
    if isinstance(callable, types.MethodType):
        self_name = spec.args[0]
        bindings.pop(self_name, None)
    if spec.varargs:
        bindings.pop(spec.varargs, None)
    if spec.varkw:
        bindings.pop(spec.varkw, None)
    for k, v in list(bindings.items()):
        if _is_specialization(v, Annotated):
            v, metadata = (v.__origin__, v.__metadata__)
            bindings[k] = v
        else:
            metadata = tuple()
        if only_explicit_bindings and _inject_marker not in metadata or _noinject_marker in metadata:
            del bindings[k]
        elif _is_specialization(v, Union) or _is_new_union_type(v):
            union_members = v.__args__
            new_members = tuple(set(union_members) - {type(None)})
            new_union = Union[new_members]
            union_metadata = {metadata for member in new_members for metadata in getattr(member, '__metadata__', tuple()) if _is_specialization(member, Annotated)}
            if only_explicit_bindings and _inject_marker not in union_metadata or _noinject_marker in union_metadata:
                del bindings[k]
            else:
                bindings[k] = new_union
    return bindings