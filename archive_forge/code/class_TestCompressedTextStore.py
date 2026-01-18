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
class TestCompressedTextStore(TestCaseInTempDir, TestStores):

    def get_store(self, path='.'):
        t = transport.get_transport_from_path(path)
        return TextStore(t, compressed=True)

    def test_total_size(self):
        store = self.get_store('.')
        store.register_suffix('dsc')
        store.add(BytesIO(b'goodbye'), b'123123')
        store.add(BytesIO(b'goodbye2'), b'123123', 'dsc')
        self.assertEqual(store.total_size(), (2, 55))

    def test__relpath_suffixed(self):
        my_store = TextStore(MockTransport(), prefixed=True, compressed=True)
        my_store.register_suffix('dsc')
        self.assertEqual('45/foo.dsc', my_store._relpath(b'foo', ['dsc']))