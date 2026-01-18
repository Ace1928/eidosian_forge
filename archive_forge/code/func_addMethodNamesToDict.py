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
def addMethodNamesToDict(classObj, dict, prefix, baseClass=None):
    """
    This goes through C{classObj} (and its bases) and puts method names
    starting with 'prefix' in 'dict' with a value of 1. if baseClass isn't
    None, methods will only be added if classObj is-a baseClass

    If the class in question has the methods 'prefix_methodname' and
    'prefix_methodname2', the resulting dict should look something like:
    {"methodname": 1, "methodname2": 1}.

    @param classObj: A class object from which to collect method names.

    @param dict: A L{dict} which will be updated with the results of the
        accumulation.  Items are added to this dictionary, with method names as
        keys and C{1} as values.
    @type dict: L{dict}

    @param prefix: A native string giving a prefix.  Each method of C{classObj}
        (and base classes of C{classObj}) with a name which begins with this
        prefix will be returned.
    @type prefix: L{str}

    @param baseClass: A class object at which to stop searching upwards for new
        methods.  To collect all method names, do not pass a value for this
        parameter.

    @return: L{None}
    """
    for base in classObj.__bases__:
        addMethodNamesToDict(base, dict, prefix, baseClass)
    if baseClass is None or baseClass in classObj.__bases__:
        for name, method in classObj.__dict__.items():
            optName = name[len(prefix):]
            if type(method) is types.FunctionType and name[:len(prefix)] == prefix and len(optName):
                dict[optName] = 1