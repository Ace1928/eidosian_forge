from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class MockResettableResource(MockResource):
    """Mock resource that logs the number of reset calls too."""

    def __init__(self):
        super(MockResettableResource, self).__init__()
        self.resets = 0

    def _reset(self, resource, dependency_resources):
        self.resets += 1
        resource._name += '!'
        self._dirty = False
        return resource