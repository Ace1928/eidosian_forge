import inspect
import unittest
from traits.api import (
class MyNewCallable2(HasTraits):
    value = Callable(pow, allow_none=True)
    empty_callable = Callable()
    a_non_none_union = Union(Callable(allow_none=False), Int)
    a_allow_none_union = Union(Callable(allow_none=True), Int)