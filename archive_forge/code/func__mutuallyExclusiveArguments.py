from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def _mutuallyExclusiveArguments(argumentPairs):
    """
    Decorator which causes its decoratee to raise a L{TypeError} if two of the
    given arguments are passed at the same time.

    @param argumentPairs: pairs of argument identifiers, each pair indicating
        an argument that may not be passed in conjunction with another.
    @type argumentPairs: sequence of 2-sequences of L{str}

    @return: A decorator, used like so::

            @_mutuallyExclusiveArguments([["tweedledum", "tweedledee"]])
            def function(tweedledum=1, tweedledee=2):
                "Don't pass tweedledum and tweedledee at the same time."

    @rtype: 1-argument callable taking a callable and returning a callable.
    """

    def wrapper(wrappee):
        spec = inspect.signature(wrappee)
        _passed = _passedSignature

        @wraps(wrappee)
        def wrapped(*args, **kwargs):
            arguments = _passed(spec, args, kwargs)
            for this, that in argumentPairs:
                if this in arguments and that in arguments:
                    raise TypeError('The %r and %r arguments to %s are mutually exclusive.' % (this, that, _fullyQualifiedName(wrappee)))
            return wrappee(*args, **kwargs)
        return wrapped
    return wrapper