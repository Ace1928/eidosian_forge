import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class MakeCounter(testresources.TestResource):
    """Test resource that counts makes and cleans."""

    def __init__(self):
        testresources.TestResource.__init__(self)
        self.cleans = 0
        self.makes = 0
        self.calls = []

    def clean(self, resource):
        self.cleans += 1
        self.calls.append(('clean', resource))

    def make(self, dependency_resources):
        self.makes += 1
        resource = 'boo %d' % self.makes
        self.calls.append(('make', resource))
        return resource