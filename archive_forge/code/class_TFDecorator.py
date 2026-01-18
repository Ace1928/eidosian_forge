import inspect
from typing import Dict, Any
class TFDecorator(object):
    """Base class for all TensorFlow decorators.

  TFDecorator captures and exposes the wrapped target, and provides details
  about the current decorator.
  """

    def __init__(self, decorator_name, target, decorator_doc='', decorator_argspec=None):
        self._decorated_target = target
        self._decorator_name = decorator_name
        self._decorator_doc = decorator_doc
        self._decorator_argspec = decorator_argspec
        if hasattr(target, '__name__'):
            self.__name__ = target.__name__
        if hasattr(target, '__qualname__'):
            self.__qualname__ = target.__qualname__
        if self._decorator_doc:
            self.__doc__ = self._decorator_doc
        elif hasattr(target, '__doc__') and target.__doc__:
            self.__doc__ = target.__doc__
        else:
            self.__doc__ = ''
        if decorator_argspec:
            self.__signature__ = fullargspec_to_signature(decorator_argspec)
        elif callable(target):
            try:
                self.__signature__ = inspect.signature(target)
            except (TypeError, ValueError):
                pass

    def __get__(self, instance, owner):
        return self._decorated_target.__get__(instance, owner)

    def __call__(self, *args, **kwargs):
        return self._decorated_target(*args, **kwargs)

    @property
    def decorated_target(self):
        return self._decorated_target

    @decorated_target.setter
    def decorated_target(self, decorated_target):
        self._decorated_target = decorated_target

    @property
    def decorator_name(self):
        return self._decorator_name

    @property
    def decorator_doc(self):
        return self._decorator_doc

    @property
    def decorator_argspec(self):
        return self._decorator_argspec