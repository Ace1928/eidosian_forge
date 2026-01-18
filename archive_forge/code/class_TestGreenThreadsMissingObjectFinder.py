import time
from dulwich.tests import TestCase, skipIf
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, parse_timezone
@skipIf(not gevent_support, skipmsg)
class TestGreenThreadsMissingObjectFinder(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.cmt_amount = 10
        self.objs = init_store(self.store, self.cmt_amount)

    def test_finder(self):
        wants = [sha.id for sha in self.objs if isinstance(sha, Commit)]
        finder = GreenThreadsMissingObjectFinder(self.store, (), wants)
        self.assertEqual(len(finder.sha_done), 0)
        self.assertEqual(len(finder.objects_to_send), self.cmt_amount)
        finder = GreenThreadsMissingObjectFinder(self.store, wants[0:int(self.cmt_amount / 2)], wants)
        self.assertEqual(len(finder.sha_done), self.cmt_amount / 2 * 2)
        self.assertEqual(len(finder.objects_to_send), self.cmt_amount / 2)