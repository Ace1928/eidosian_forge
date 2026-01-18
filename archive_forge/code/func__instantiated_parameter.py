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
def _instantiated_parameter(parameterized, param):
    """
    Given a Parameterized object and one of its class Parameter objects,
    return the appropriate Parameter object for this instance, instantiating
    it if need be.
    """
    if getattr(parameterized._param__private, 'initialized', False) and param.per_instance and (not getattr(type(parameterized)._param__private, 'disable_instance_params', False)):
        key = param.name
        if key not in parameterized._param__private.params:
            parameterized._param__private.params[key] = _instantiate_param_obj(param, parameterized)
        param = parameterized._param__private.params[key]
    return param