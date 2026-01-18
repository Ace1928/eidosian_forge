import gzip
import os
from io import BytesIO
from ... import errors as errors
from ... import transactions, transport
from ...bzr.weave import WeaveFile
from ...errors import BzrError
from ...tests import TestCase, TestCaseInTempDir, TestCaseWithTransport
from ...transport.memory import MemoryTransport
from .store import TransportStore
from .store.text import TextStore
from .store.versioned import VersionedFileStore
class TestMemoryStore(TestCase):

    def get_store(self):
        return TextStore(MemoryTransport())

    def test_add_and_retrieve(self):
        store = self.get_store()
        store.add(BytesIO(b'hello'), b'aa')
        self.assertNotEqual(store.get(b'aa'), None)
        self.assertEqual(store.get(b'aa').read(), b'hello')
        store.add(BytesIO(b'hello world'), b'bb')
        self.assertNotEqual(store.get(b'bb'), None)
        self.assertEqual(store.get(b'bb').read(), b'hello world')

    def test_missing_is_absent(self):
        store = self.get_store()
        self.assertNotIn(b'aa', store)

    def test_adding_fails_when_present(self):
        my_store = self.get_store()
        my_store.add(BytesIO(b'hello'), b'aa')
        self.assertRaises(BzrError, my_store.add, BytesIO(b'hello'), b'aa')

    def test_total_size(self):
        store = self.get_store()
        store.add(BytesIO(b'goodbye'), b'123123')
        store.add(BytesIO(b'goodbye2'), b'123123.dsc')
        self.assertEqual(store.total_size(), (2, 15))