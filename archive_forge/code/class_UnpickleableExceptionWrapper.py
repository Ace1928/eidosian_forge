import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
class UnpickleableExceptionWrapper(Exception):
    """Wraps unpickleable exceptions.

    Arguments:
        exc_module (str): See :attr:`exc_module`.
        exc_cls_name (str): See :attr:`exc_cls_name`.
        exc_args (Tuple[Any, ...]): See :attr:`exc_args`.

    Example:
        >>> def pickle_it(raising_function):
        ...     try:
        ...         raising_function()
        ...     except Exception as e:
        ...         exc = UnpickleableExceptionWrapper(
        ...             e.__class__.__module__,
        ...             e.__class__.__name__,
        ...             e.args,
        ...         )
        ...         pickle.dumps(exc)  # Works fine.
    """
    exc_module = None
    exc_cls_name = None
    exc_args = None

    def __init__(self, exc_module, exc_cls_name, exc_args, text=None):
        safe_exc_args = ensure_serializable(exc_args, lambda v: pickle.loads(pickle.dumps(v)))
        self.exc_module = exc_module
        self.exc_cls_name = exc_cls_name
        self.exc_args = safe_exc_args
        self.text = text
        super().__init__(exc_module, exc_cls_name, safe_exc_args, text)

    def restore(self):
        return create_exception_cls(self.exc_cls_name, self.exc_module)(*self.exc_args)

    def __str__(self):
        return self.text

    @classmethod
    def from_exception(cls, exc):
        res = cls(exc.__class__.__module__, exc.__class__.__name__, getattr(exc, 'args', []), safe_repr(exc))
        if hasattr(exc, '__traceback__'):
            res = res.with_traceback(exc.__traceback__)
        return res