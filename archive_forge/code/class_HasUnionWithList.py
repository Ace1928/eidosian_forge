import unittest
from traits.api import (
class HasUnionWithList(HasTraits):
    foo = Union(Int(23), Float)
    nested = Union(Union(Str(), Bytes()), Union(Int(), Float(), None))