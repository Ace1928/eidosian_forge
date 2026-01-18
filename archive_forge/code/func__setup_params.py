import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
@as_uninitialized
def _setup_params(self_, **params):
    """
        Initialize default and keyword parameter values.

        First, ensures that values for all Parameters with 'instantiate=True'
        (typically used for mutable Parameters) are copied directly into each object,
        to ensure that there is an independent copy of the value (to avoid surprising
        aliasing errors). Second, ensures that Parameters with 'constant=True' are
        referenced on the instance, to make sure that setting a constant
        Parameter on the class doesn't affect already created instances. Then
        sets each of the keyword arguments, raising when any of them are not
        defined as parameters.
        """
    self = self_.self
    params_to_deepcopy = {}
    params_to_ref = {}
    objects = self_._cls_parameters
    for pname, p in objects.items():
        if p.instantiate and pname != 'name':
            params_to_deepcopy[pname] = p
        elif p.constant and pname != 'name':
            params_to_ref[pname] = p
    for p in params_to_deepcopy.values():
        self_._instantiate_param(p)
    for p in params_to_ref.values():
        self_._instantiate_param(p, deepcopy=False)
    deps, refs = ({}, {})
    for name, val in params.items():
        desc = self_.cls.get_param_descriptor(name)[0]
        if not desc:
            raise TypeError(f'{self.__class__.__name__}.__init__() got an unexpected keyword argument {name!r}')
        pobj = objects.get(name)
        if pobj is None or not pobj.allow_refs:
            if name not in self_.cls._param__private.explicit_no_refs:
                try:
                    ref, _, resolved, _ = self_._resolve_ref(pobj, val)
                except Exception:
                    ref = None
                if ref:
                    warnings.warn(f'Parameter {name!r} on {pobj.owner} is being given a valid parameter reference {val} but is implicitly allow_refs=False. In future allow_refs will be enabled by default and the reference {val} will be resolved to its underlying value {resolved}. Please explicitly set allow_ref on the Parameter definition to declare whether references should be resolved or not.', category=_ParamFutureWarning, stacklevel=4)
            setattr(self, name, val)
            continue
        ref, ref_deps, resolved, is_async = self_._resolve_ref(pobj, val)
        if ref is not None:
            refs[name] = ref
            deps[name] = ref_deps
        if not is_async and (not (resolved is Undefined or resolved is Skip)):
            setattr(self, name, resolved)
    return (refs, deps)