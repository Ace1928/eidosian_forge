from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
class FakeRootTest(ResourcedTestCase):

    def test_create_root_resource(self):
        root_resource = FakeRoot(get_application())
        self.assertTrue(isinstance(root_resource, FakeResource))