import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
def _get_named_tuple_properties(obj, loc: Optional[torch._C._jit_tree_views.SourceRange]=None, rcb=None):
    if loc is None:
        loc = fake_range()
    assert issubclass(obj, tuple) and hasattr(obj, '_fields')
    if hasattr(obj, '_field_defaults'):
        defaults = [obj._field_defaults[field] for field in obj._fields if field in obj._field_defaults]
    else:
        defaults = []
    if sys.version_info[:2] < (3, 10):
        obj_annotations = getattr(obj, '__annotations__', {})
    else:
        obj_annotations = inspect.get_annotations(obj)
        if len(obj_annotations) == 0 and hasattr(obj, '__base__'):
            obj_annotations = inspect.get_annotations(obj.__base__)
    annotations = []
    for field in obj._fields:
        if field in obj_annotations:
            field_type = obj_annotations[field]
            if isinstance(field_type, ForwardRef) and rcb is not None:
                rcb_type = rcb(field_type.__forward_arg__)
                if rcb_type is None:
                    raise ValueError(f"Unknown type annotation: '{field_type}' in NamedTuple {obj.__name__}. Likely due to partial support for ForwardRef parameters in NamedTuples, see #95858. Issue occurred at {loc.highlight()}")
                field_type = rcb_type
            the_type = torch.jit.annotations.ann_to_type(field_type, loc, rcb)
            annotations.append(the_type)
        else:
            annotations.append(torch._C.TensorType.getInferred())
    return (type(obj).__name__, obj._fields, annotations, defaults)