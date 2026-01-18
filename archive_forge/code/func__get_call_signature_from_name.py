import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
def _get_call_signature_from_name(self, name):
    """
        * If call objects are asserted against a method/function like obj.meth1
        then there could be no name for the call object to lookup. Hence just
        return the spec_signature of the method/function being asserted against.
        * If the name is not empty then remove () and split by '.' to get
        list of names to iterate through the children until a potential
        match is found. A child mock is created only during attribute access
        so if we get a _SpecState then no attributes of the spec were accessed
        and can be safely exited.
        """
    if not name:
        return self._spec_signature
    sig = None
    names = name.replace('()', '').split('.')
    children = self._mock_children
    for name in names:
        child = children.get(name)
        if child is None or isinstance(child, _SpecState):
            break
        else:
            child = _extract_mock(child)
            children = child._mock_children
            sig = child._spec_signature
    return sig