import functools
import io
import logging
import os
import sys
import time
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from oslo_utils import timeutils
class exception_filter(object):
    """A context manager that prevents some exceptions from being raised.

    Use this class as a decorator for a function that returns whether a given
    exception should be ignored, in cases where complex logic beyond subclass
    matching is required. e.g.

    >>> @exception_filter
    >>> def ignore_test_assertions(ex):
    ...     return isinstance(ex, AssertionError) and 'test' in str(ex)

    The filter matching function can then be used as a context manager:

    >>> with ignore_test_assertions:
    ...     assert False, 'This is a test'

    or called directly:

    >>> try:
    ...     assert False, 'This is a test'
    ... except Exception as ex:
    ...     ignore_test_assertions(ex)

    Any non-matching exception will be re-raised. When the filter is used as a
    context manager, the traceback for re-raised exceptions is always
    preserved. When the filter is called as a function, the traceback is
    preserved provided that no other exceptions have been raised in the
    intervening time. The context manager method is preferred for this reason
    except in cases where the ignored exception affects control flow.
    """

    def __init__(self, should_ignore_ex):
        self._should_ignore_ex = should_ignore_ex
        if all((hasattr(should_ignore_ex, a) for a in functools.WRAPPER_ASSIGNMENTS)):
            functools.update_wrapper(self, should_ignore_ex)

    def __get__(self, obj, owner):
        return type(self)(self._should_ignore_ex.__get__(obj, owner))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            return self._should_ignore_ex(exc_val)

    def __call__(self, ex):
        """Re-raise any exception value not being filtered out.

        If the exception was the last to be raised, it will be re-raised with
        its original traceback.
        """
        exc_type, exc_val, traceback = sys.exc_info()
        try:
            if not self._should_ignore_ex(ex):
                if exc_val is ex:
                    try:
                        if exc_val is None:
                            exc_val = exc_type()
                        if exc_val.__traceback__ is not traceback:
                            raise exc_val.with_traceback(traceback)
                        raise exc_val
                    finally:
                        exc_val = None
                        traceback = None
                else:
                    raise ex
        finally:
            del exc_type, exc_val, traceback