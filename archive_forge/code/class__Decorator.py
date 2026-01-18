from __future__ import annotations
from functools import wraps
import zmq
class _Decorator:
    """The mini decorator factory"""

    def __init__(self, target=None):
        self._target = target

    def __call__(self, *dec_args, **dec_kwargs):
        """
        The main logic of decorator

        Here is how those arguments works::

            @out_decorator(*dec_args, *dec_kwargs)
            def func(*wrap_args, **wrap_kwargs):
                ...

        And in the ``wrapper``, we simply create ``self.target`` instance via
        ``with``::

            target = self.get_target(*args, **kwargs)
            with target(*dec_args, **dec_kwargs) as obj:
                ...

        """
        kw_name, dec_args, dec_kwargs = self.process_decorator_args(*dec_args, **dec_kwargs)

        def decorator(func):

            @wraps(func)
            def wrapper(*args, **kwargs):
                target = self.get_target(*args, **kwargs)
                with target(*dec_args, **dec_kwargs) as obj:
                    if kw_name and kw_name not in kwargs:
                        kwargs[kw_name] = obj
                    elif kw_name and kw_name in kwargs:
                        raise TypeError(f"{func.__name__}() got multiple values for argument '{kw_name}'")
                    else:
                        args = args + (obj,)
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def get_target(self, *args, **kwargs):
        """Return the target function

        Allows modifying args/kwargs to be passed.
        """
        return self._target

    def process_decorator_args(self, *args, **kwargs):
        """Process args passed to the decorator.

        args not consumed by the decorator will be passed to the target factory
        (Context/Socket constructor).
        """
        kw_name = None
        if isinstance(kwargs.get('name'), str):
            kw_name = kwargs.pop('name')
        elif len(args) >= 1 and isinstance(args[0], str):
            kw_name = args[0]
            args = args[1:]
        return (kw_name, args, kwargs)