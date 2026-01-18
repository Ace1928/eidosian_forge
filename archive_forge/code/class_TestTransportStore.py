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
class TestTransportStore(TestCase):

    def test__relpath_invalid(self):
        my_store = TransportStore(MockTransport())
        self.assertRaises(ValueError, my_store._relpath, b'/foo')
        self.assertRaises(ValueError, my_store._relpath, b'foo/')

    def test_register_invalid_suffixes(self):
        my_store = TransportStore(MockTransport())
        self.assertRaises(ValueError, my_store.register_suffix, '/')
        self.assertRaises(ValueError, my_store.register_suffix, '.gz/bar')

    def test__relpath_unregister_suffixes(self):
        my_store = TransportStore(MockTransport())
        self.assertRaises(ValueError, my_store._relpath, b'foo', [b'gz'])
        self.assertRaises(ValueError, my_store._relpath, b'foo', [b'dsc', b'gz'])

    def test__relpath_simple(self):
        my_store = TransportStore(MockTransport())
        self.assertEqual('foo', my_store._relpath(b'foo'))

    def test__relpath_prefixed(self):
        my_store = TransportStore(MockTransport(), True)
        self.assertEqual('45/foo', my_store._relpath(b'foo'))

    def test__relpath_simple_suffixed(self):
        my_store = TransportStore(MockTransport())
        my_store.register_suffix('bar')
        my_store.register_suffix('baz')
        self.assertEqual('foo.baz', my_store._relpath(b'foo', ['baz']))
        self.assertEqual('foo.bar.baz', my_store._relpath(b'foo', ['bar', 'baz']))

    def test__relpath_prefixed_suffixed(self):
        my_store = TransportStore(MockTransport(), True)
        my_store.register_suffix('bar')
        my_store.register_suffix('baz')
        self.assertEqual('45/foo.baz', my_store._relpath(b'foo', ['baz']))
        self.assertEqual('45/foo.bar.baz', my_store._relpath(b'foo', ['bar', 'baz']))

    def test_add_simple(self):
        stream = BytesIO(b'content')
        my_store = InstrumentedTransportStore(MockTransport())
        my_store.add(stream, b'foo')
        self.assertEqual([('_add', 'foo', stream)], my_store._calls)

    def test_add_prefixed(self):
        stream = BytesIO(b'content')
        my_store = InstrumentedTransportStore(MockTransport(), True)
        my_store.add(stream, b'foo')
        self.assertEqual([('_add', '45/foo', stream)], my_store._calls)

    def test_add_simple_suffixed(self):
        stream = BytesIO(b'content')
        my_store = InstrumentedTransportStore(MockTransport())
        my_store.register_suffix('dsc')
        my_store.add(stream, b'foo', 'dsc')
        self.assertEqual([('_add', 'foo.dsc', stream)], my_store._calls)

    def test_add_simple_suffixed_dir(self):
        stream = BytesIO(b'content')
        my_store = InstrumentedTransportStore(MockTransport(), True)
        my_store.register_suffix('dsc')
        my_store.add(stream, b'foo', 'dsc')
        self.assertEqual([('_add', '45/foo.dsc', stream)], my_store._calls)

    def get_populated_store(self, prefixed=False, store_class=TextStore, compressed=False):
        my_store = store_class(MemoryTransport(), prefixed, compressed=compressed)
        my_store.register_suffix('sig')
        stream = BytesIO(b'signature')
        my_store.add(stream, b'foo', 'sig')
        stream = BytesIO(b'content')
        my_store.add(stream, b'foo')
        stream = BytesIO(b'signature for missing base')
        my_store.add(stream, b'missing', 'sig')
        return my_store

    def test_has_simple(self):
        my_store = self.get_populated_store()
        self.assertEqual(True, my_store.has_id(b'foo'))
        my_store = self.get_populated_store(True)
        self.assertEqual(True, my_store.has_id(b'foo'))

    def test_has_suffixed(self):
        my_store = self.get_populated_store()
        self.assertEqual(True, my_store.has_id(b'foo', 'sig'))
        my_store = self.get_populated_store(True)
        self.assertEqual(True, my_store.has_id(b'foo', 'sig'))

    def test_has_suffixed_no_base(self):
        my_store = self.get_populated_store()
        self.assertEqual(False, my_store.has_id(b'missing'))
        my_store = self.get_populated_store(True)
        self.assertEqual(False, my_store.has_id(b'missing'))

    def test_get_simple(self):
        my_store = self.get_populated_store()
        self.assertEqual(b'content', my_store.get(b'foo').read())
        my_store = self.get_populated_store(True)
        self.assertEqual(b'content', my_store.get(b'foo').read())

    def test_get_suffixed(self):
        my_store = self.get_populated_store()
        self.assertEqual(b'signature', my_store.get(b'foo', 'sig').read())
        my_store = self.get_populated_store(True)
        self.assertEqual(b'signature', my_store.get(b'foo', 'sig').read())

    def test_get_suffixed_no_base(self):
        my_store = self.get_populated_store()
        self.assertEqual(b'signature for missing base', my_store.get(b'missing', 'sig').read())
        my_store = self.get_populated_store(True)
        self.assertEqual(b'signature for missing base', my_store.get(b'missing', 'sig').read())

    def test___iter__no_suffix(self):
        my_store = TextStore(MemoryTransport(), prefixed=False, compressed=False)
        stream = BytesIO(b'content')
        my_store.add(stream, b'foo')
        self.assertEqual({b'foo'}, set(my_store.__iter__()))

    def test___iter__(self):
        self.assertEqual({b'foo'}, set(self.get_populated_store().__iter__()))
        self.assertEqual({b'foo'}, set(self.get_populated_store(True).__iter__()))

    def test___iter__compressed(self):
        self.assertEqual({b'foo'}, set(self.get_populated_store(compressed=True).__iter__()))
        self.assertEqual({b'foo'}, set(self.get_populated_store(True, compressed=True).__iter__()))

    def test___len__(self):
        self.assertEqual(1, len(self.get_populated_store()))

    def test_relpath_escaped(self):
        my_store = TransportStore(MemoryTransport())
        self.assertEqual('%25', my_store._relpath(b'%'))

    def test_escaped_uppercase(self):
        """Uppercase letters are escaped for safety on Windows"""
        my_store = TransportStore(MemoryTransport(), prefixed=True, escaped=True)
        self.assertEqual(my_store._relpath(b'C:<>'), 'be/%2543%253a%253c%253e')