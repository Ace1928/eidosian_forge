import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
def _named_objs(objlist, namesdict=None):
    """
    Given a list of objects, returns a dictionary mapping from
    string name for the object to the object itself. Accepts
    an optional name,obj dictionary, which will override any other
    name if that item is present in the dictionary.
    """
    objs = OrderedDict()
    objtoname = {}
    unhashables = []
    if namesdict is not None:
        for k, v in namesdict.items():
            try:
                objtoname[_hashable(v)] = k
            except TypeError:
                unhashables.append((k, v))
    for obj in objlist:
        if objtoname and _hashable(obj) in objtoname:
            k = objtoname[_hashable(obj)]
        elif any((obj is v for _, v in unhashables)):
            k = [k for k, v in unhashables if v is obj][0]
        elif hasattr(obj, 'name'):
            k = obj.name
        elif hasattr(obj, '__name__'):
            k = obj.__name__
        else:
            k = str(obj)
        objs[k] = obj
    return objs