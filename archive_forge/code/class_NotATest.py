import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
class NotATest:

    def test_foo(self) -> None:
        pass