from dulwich.objects import Blob
from dulwich.tests.test_object_store import PackBasedObjectStoreTests
from dulwich.tests.utils import make_object
from ...tests import TestCaseWithTransport
from ..transportgit import TransportObjectStore, TransportRefsContainer
class TransportRefContainerTests(TestCaseWithTransport):

    def setUp(self):
        TestCaseWithTransport.setUp(self)
        self._refs = TransportRefsContainer(self.get_transport())

    def test_packed_refs_missing(self):
        self.assertEqual({}, self._refs.get_packed_refs())

    def test_packed_refs(self):
        self.get_transport().put_bytes_non_atomic('packed-refs', b'# pack-refs with: peeled fully-peeled sorted \n2001b954f1ec392f84f7cec2f2f96a76ed6ba4ee refs/heads/master')
        self.assertEqual({b'refs/heads/master': b'2001b954f1ec392f84f7cec2f2f96a76ed6ba4ee'}, self._refs.get_packed_refs())