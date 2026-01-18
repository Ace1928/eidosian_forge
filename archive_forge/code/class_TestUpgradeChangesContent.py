from ....tests import TestCase
from ..upgrade import UpgradeChangesContent
class TestUpgradeChangesContent(TestCase):

    def test_init(self):
        x = UpgradeChangesContent('revisionx')
        self.assertEqual('revisionx', x.revid)