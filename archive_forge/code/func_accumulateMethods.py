import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
def accumulateMethods(obj, dict, prefix='', curClass=None):
    """
    Given an object C{obj}, add all methods that begin with C{prefix}.

    @param obj: An arbitrary object to collect methods from.

    @param dict: A L{dict} which will be updated with the results of the
        accumulation.  Items are added to this dictionary, with method names as
        keys and corresponding instance method objects as values.
    @type dict: L{dict}

    @param prefix: A native string giving a prefix.  Each method of C{obj} with
        a name which begins with this prefix will be returned.
    @type prefix: L{str}

    @param curClass: The class in the inheritance hierarchy at which to start
        collecting methods.  Collection proceeds up.  To collect all methods
        from C{obj}, do not pass a value for this parameter.

    @return: L{None}
    """
    if not curClass:
        curClass = obj.__class__
    for base in curClass.__bases__:
        if base is not object:
            accumulateMethods(obj, dict, prefix, base)
    for name, method in curClass.__dict__.items():
        optName = name[len(prefix):]
        if type(method) is types.FunctionType and name[:len(prefix)] == prefix and len(optName):
            dict[optName] = getattr(obj, name)