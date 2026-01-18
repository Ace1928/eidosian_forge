from cinderclient import extension
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.contrib import list_extensions
class ListExtensionsTests(utils.TestCase):

    def test_list_extensions(self):
        all_exts = cs.list_extensions.show_all()
        cs.assert_called('GET', '/extensions')
        self.assertGreater(len(all_exts), 0)
        for r in all_exts:
            self.assertGreater(len(r.summary), 0)