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
def __set_name(mcs, name, dict_):
    """
        Give Parameterized classes a useful 'name' attribute that is by
        default the class name, unless a class in the hierarchy has defined
        a `name` String Parameter with a defined `default` value, in which case
        that value is used to set the class name.
        """
    name_param = dict_.get('name', None)
    if name_param is not None:
        if not type(name_param) is String:
            raise TypeError(f"Parameterized class {name!r} cannot override the 'name' Parameter with type {type(name_param)}. Overriding 'name' is only allowed with a 'String' Parameter.")
        if name_param.default:
            mcs.name = name_param.default
            mcs._param__private.renamed = True
        else:
            mcs.name = name
    else:
        classes = classlist(mcs)[::-1]
        found_renamed = False
        for c in classes:
            if hasattr(c, '_param__private') and c._param__private.renamed:
                found_renamed = True
                break
        if not found_renamed:
            mcs.name = name