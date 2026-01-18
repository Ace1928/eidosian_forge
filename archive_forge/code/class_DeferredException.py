from . import version
import collections
from functools import wraps
import sys
import warnings
class DeferredException(object):
    """Allows to defer exceptions."""

    def __init__(self, type=None, value=None, traceback=None):
        """Return a new DeferredException instance.

        If type, value and traceback are not specified the infotmation
        will be retreieved from the last caught exception:

        >>> try:
        ...     raise Exception("Test")
        ... except:
        ...     deferred_exc = DeferredException()
        >>> deferred_exc.raise_exception()
        Traceback (most recent call last):
            ...
        Exception: Test

        Alternatively you can set the exception manually:

        >>> exception = Exception("Test 2")
        >>> deferred_exc = DeferredException(exception)
        >>> deferred_exc.raise_exception()
        Traceback (most recent call last):
            ...
        Exception: Test 2
        """
        self.type = type
        self.value = value
        self.traceback = traceback
        if isinstance(type, Exception):
            self.type = type.__class__
            self.value = type
        elif not type or not value:
            self.type, self.value, self.traceback = sys.exc_info()

    def raise_exception(self):
        """Raise the stored exception."""
        raise self.type(self.value).with_traceback(self.traceback)

    def catch(self, *errors):
        """Check if the stored exception is a subclass of one of the
        provided exception classes. If this is the case return the
        matching exception class. Otherwise raise the stored exception.

        >>> exc = DeferredException(SystemError())
        >>> exc.catch(Exception) # Will catch the exception and return it
        <type 'exceptions.Exception'>
        >>> exc.catch(OSError)   # Won't catch and raise the stored exception
        Traceback (most recent call last):
            ...
        SystemError

        This method can be used in errbacks of a Deferred:

        >>> def dummy_errback(deferred_exception):
        ...     '''Error handler for OSError'''
        ...     deferred_exception.catch(OSError)
        ...     return "catched"

        The above errback can handle an OSError:

        >>> deferred = Deferred()
        >>> deferred.add_errback(dummy_errback)
        >>> deferred.errback(OSError())
        >>> deferred.result
        'catched'

        But fails to handle a SystemError:

        >>> deferred2 = Deferred()
        >>> deferred2.add_errback(dummy_errback)
        >>> deferred2.errback(SystemError())
        >>> deferred2.result                             #doctest: +ELLIPSIS
        <defer.DeferredException object at 0x...>
        >>> deferred2.result.value
        SystemError()
        """
        for err in errors:
            if issubclass(self.type, err):
                return err
        self.raise_exception()