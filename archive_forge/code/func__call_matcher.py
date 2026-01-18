from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _call_matcher(self, _call):
    """
        Given a call (or simply a (args, kwargs) tuple), return a
        comparison key suitable for matching with other calls.
        This is a best effort method which relies on the spec's signature,
        if available, or falls back on the arguments themselves.
        """
    sig = self._spec_signature
    if sig is not None:
        if len(_call) == 2:
            name = ''
            args, kwargs = _call
        else:
            name, args, kwargs = _call
        try:
            return (name, sig.bind(*args, **kwargs))
        except TypeError as e:
            e.__traceback__ = None
            return e
    else:
        return _call