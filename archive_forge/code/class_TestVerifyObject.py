import unittest
from zope.interface.common import builtins
from . import VerifyClassMixin
from . import VerifyObjectMixin
from . import add_verify_tests
class TestVerifyObject(VerifyObjectMixin, TestVerifyClass):
    CONSTRUCTORS = {builtins.IFile: lambda: open(__file__)}