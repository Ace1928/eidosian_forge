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
class _ParameterBase(metaclass=ParameterMetaclass):
    """
    Base Parameter class used to dynamically update the signature of all
    the Parameters.
    """

    @classmethod
    def _modified_slots_defaults(cls):
        defaults = cls._slot_defaults.copy()
        defaults['label'] = defaults.pop('_label')
        return defaults

    @classmethod
    def __init_subclass__(cls):
        try:
            cls._update_signature()
        except Exception:
            cls.__signature__ = inspect.signature(cls.__init__)

    @classmethod
    def _update_signature(cls):
        defaults = cls._modified_slots_defaults()
        new_parameters = {}
        for i, kls in enumerate(cls.mro()):
            if kls.__name__.startswith('_'):
                continue
            sig = inspect.signature(kls.__init__)
            for pname, parameter in sig.parameters.items():
                if pname == 'self':
                    continue
                if i >= 1 and parameter.default == inspect.Signature.empty:
                    continue
                if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue
                if getattr(parameter, 'default', None) is Undefined:
                    if pname not in defaults:
                        raise LookupError(f'Argument {pname!r} of Parameter {cls.__name__!r} has no entry in _slot_defaults.')
                    default = defaults[pname]
                    if callable(default) and hasattr(default, 'sig'):
                        default = default.sig
                    new_parameter = parameter.replace(default=default)
                else:
                    new_parameter = parameter
                if i >= 1:
                    new_parameter = new_parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
                new_parameters.setdefault(pname, new_parameter)

        def _sorter(p):
            if p.default == inspect.Signature.empty:
                return 0
            else:
                return 1
        new_parameters = sorted(new_parameters.values(), key=_sorter)
        new_sig = sig.replace(parameters=new_parameters)
        cls.__signature__ = new_sig