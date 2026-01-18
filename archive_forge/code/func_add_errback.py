from . import version
import collections
from functools import wraps
import sys
import warnings
def add_errback(self, func, *args, **kwargs):
    """Add a callable (function or method) to the errback chain only.

        If there isn't any exception the result will be passed through to
        the callback of the next pair.

        The first argument is the callable instance followed by any
        additional argument that will be passed to the errback.

        The errback method will get the most recent DeferredException and
        and any additional arguments that was specified in add_errback.

        If the errback can catch the exception it can return a value that
        will be passed to the next callback in the chain. Otherwise the
        errback chain will not be processed anymore.

        See the documentation of defer.DeferredException.catch for
        further information.

        >>> def catch_error(deferred_error, ignore=False):
        ...     if ignore:
        ...         return "ignored"
        ...     deferred_error.catch(Exception)
        ...     return "catched"
        >>> deferred = Deferred()
        >>> deferred.errback(SystemError())
        >>> deferred.add_errback(catch_error, ignore=True)
        >>> deferred.result
        'ignored'
        """
    self.add_callbacks(_passthrough, func, errback_args=args, errback_kwargs=kwargs)