import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps

        Update the docstring and wrap the ``__init__`` in-place (or ``__new__``
        if the class or any of the bases overrides ``__new__``) so it will give
        a deprecation warning when an instance is created.

        This won't work for extension classes because these can't be modified
        in-place and the alternatives don't work in the general case:

        - Using a new class that looks and behaves like the original doesn't
          work because the __new__ method of extension types usually makes sure
          that it's the same class or a subclass.
        - Subclassing the class and return the subclass can lead to problems
          with pickle and will look weird in the Sphinx docs.
        