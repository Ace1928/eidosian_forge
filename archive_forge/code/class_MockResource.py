from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class MockResource(testresources.TestResourceManager):
    """Mock resource that logs the number of make and clean calls."""

    def __init__(self):
        super(MockResource, self).__init__()
        self.makes = 0
        self.cleans = 0

    def clean(self, resource):
        self.cleans += 1

    def make(self, dependency_resources):
        self.makes += 1
        return MockResourceInstance('Boo!')