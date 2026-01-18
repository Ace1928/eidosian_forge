import functools
import logging
import sys
class base_timeoutable(object):
    """A base for function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.

    Any decorated callable may receive a ``timeout`` optional parameter that
    specifies the number of seconds allocated to the callable execution.

    The decorated functions that exceed that timeout return ``None`` or the
    value provided by the decorator.

    :param default: The default value in case we timed out during the decorated
      function execution. Default is None.

    :param timeout_param: As adding dynamically a ``timeout`` named parameter
      to the decorated callable may conflict with the callable signature, you
      may choose another name to provide that parameter. Your decoration line
      could look like ``@timeoutable(timeout_param='my_timeout')``

    .. note::

       This is a base class that must be subclassed. subclasses must override
       thz ``to_ctx_mgr`` with a timeout  context manager class which in turn
       must subclasses of above ``BaseTimeout`` class.
    """
    to_ctx_mgr = None

    def __init__(self, default=None, timeout_param='timeout'):
        self.default, self.timeout_param = (default, timeout_param)

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timeout = kwargs.pop(self.timeout_param, None)
            if timeout:
                with self.to_ctx_mgr(timeout, swallow_exc=True):
                    result = self.default
                    result = func(*args, **kwargs)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper