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
class ParameterMetaclass(type):
    """
    Metaclass allowing control over creation of Parameter classes.
    """

    def __new__(mcs, classname, bases, classdict):
        if '__doc__' in classdict:
            classdict['__classdoc'] = classdict['__doc__']
        classdict['__doc__'] = property(attrgetter('doc'))
        all_slots = {}
        for bcls in set(chain(*(base.__mro__[::-1] for base in bases))):
            all_slots.update(dict.fromkeys(getattr(bcls, '__slots__', [])))
        if '__slots__' not in classdict:
            classdict['__slots__'] = []
        else:
            all_slots.update(dict.fromkeys(classdict['__slots__']))
        classdict['_all_slots_'] = list(all_slots)
        return type.__new__(mcs, classname, bases, classdict)

    def __getattribute__(mcs, name):
        if name == '__doc__':
            return type.__getattribute__(mcs, '__classdoc')
        else:
            return type.__getattribute__(mcs, name)