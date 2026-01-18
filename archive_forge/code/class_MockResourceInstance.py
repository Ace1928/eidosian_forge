from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class MockResourceInstance(object):

    def __init__(self, name):
        self._name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __cmp__(self, other):
        return cmp(self.__dict__, other.__dict__)

    def __repr__(self):
        return self._name