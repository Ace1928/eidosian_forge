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
class ParameterizedFunction(Parameterized):
    """
    Acts like a Python function, but with arguments that are Parameters.

    Implemented as a subclass of Parameterized that, when instantiated,
    automatically invokes __call__ and returns the result, instead of
    returning an instance of the class.

    To obtain an instance of this class, call instance().
    """
    __abstract = True

    def __str__(self):
        return self.__class__.__name__ + '()'

    @bothmethod
    def instance(self_or_cls, **params):
        """
        Return an instance of this class, copying parameters from any
        existing instance provided.
        """
        if isinstance(self_or_cls, ParameterizedMetaclass):
            cls = self_or_cls
        else:
            p = params
            params = self_or_cls.param.values()
            params.update(p)
            params.pop('name')
            cls = self_or_cls.__class__
        inst = Parameterized.__new__(cls)
        Parameterized.__init__(inst, **params)
        if 'name' in params:
            inst.__name__ = params['name']
        else:
            inst.__name__ = self_or_cls.name
        return inst

    def __new__(class_, *args, **params):
        inst = class_.instance()
        inst.param._set_name(class_.__name__)
        return inst.__call__(*args, **params)

    def __call__(self, *args, **kw):
        raise NotImplementedError('Subclasses must implement __call__.')

    def __reduce__(self):
        state = ParameterizedFunction.__getstate__(self)
        return (_new_parameterized, (self.__class__,), state)

    def _pprint(self, imports=None, prefix='\n    ', unknown_value='<?>', qualify=False, separator=''):
        """
        Same as self.param.pprint, except that X.classname(Y
        is replaced with X.classname.instance(Y
        """
        r = self.param.pprint(imports, prefix, unknown_value=unknown_value, qualify=qualify, separator=separator)
        classname = self.__class__.__name__
        return r.replace('.%s(' % classname, '.%s.instance(' % classname)