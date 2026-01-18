from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class _ArgumentParserCache(type):
    """Metaclass used to cache and share argument parsers among flags."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Returns an instance of the argument parser cls.

    This method overrides behavior of the __new__ methods in
    all subclasses of ArgumentParser (inclusive). If an instance
    for cls with the same set of arguments exists, this instance is
    returned, otherwise a new instance is created.

    If any keyword arguments are defined, or the values in args
    are not hashable, this method always returns a new instance of
    cls.

    Args:
      *args: Positional initializer arguments.
      **kwargs: Initializer keyword arguments.

    Returns:
      An instance of cls, shared or new.
    """
        if kwargs:
            return type.__call__(cls, *args, **kwargs)
        else:
            instances = cls._instances
            key = (cls,) + tuple(args)
            try:
                return instances[key]
            except KeyError:
                return instances.setdefault(key, type.__call__(cls, *args))
            except TypeError:
                return type.__call__(cls, *args)