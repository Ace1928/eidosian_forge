import abc
import functools
from typing import cast, Callable, Set, TypeVar
A metaclass extending `abc.ABCMeta` for defining flexible abstract base classes

    This metadata allows the declaration of an abstract base classe (ABC)
    with more flexibility in which methods must be overridden.

    Use this metaclass in the same way as `abc.ABCMeta` to create an ABC.

    In addition to the decorators in the` abc` module, the decorator
    `@alternative(...)` may be used.
    