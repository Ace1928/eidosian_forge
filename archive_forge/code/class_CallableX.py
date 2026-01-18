import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
class CallableX(X):

    def __call__(self):
        pass