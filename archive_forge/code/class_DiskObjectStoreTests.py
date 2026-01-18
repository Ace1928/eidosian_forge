import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
class DiskObjectStoreTests(PackBasedObjectStoreTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.store_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.store_dir)
        self.store = DiskObjectStore.init(self.store_dir)

    def tearDown(self):
        TestCase.tearDown(self)
        PackBasedObjectStoreTests.tearDown(self)

    def test_loose_compression_level(self):
        alternate_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, alternate_dir)
        alternate_store = DiskObjectStore(alternate_dir, loose_compression_level=6)
        b2 = make_object(Blob, data=b'yummy data')
        alternate_store.add_object(b2)

    def test_alternates(self):
        alternate_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, alternate_dir)
        alternate_store = DiskObjectStore(alternate_dir)
        b2 = make_object(Blob, data=b'yummy data')
        alternate_store.add_object(b2)
        store = DiskObjectStore(self.store_dir)
        self.assertRaises(KeyError, store.__getitem__, b2.id)
        store.add_alternate_path(alternate_dir)
        self.assertIn(b2.id, store)
        self.assertEqual(b2, store[b2.id])

    def test_read_alternate_paths(self):
        store = DiskObjectStore(self.store_dir)
        abs_path = os.path.abspath(os.path.normpath('/abspath'))
        store.add_alternate_path(abs_path)
        self.assertEqual(set(store._read_alternate_paths()), {abs_path})
        store.add_alternate_path('relative-path')
        self.assertIn(os.path.join(store.path, 'relative-path'), set(store._read_alternate_paths()))
        store.add_alternate_path('# comment')
        for alt_path in store._read_alternate_paths():
            self.assertNotIn('#', alt_path)

    def test_file_modes(self):
        self.store.add_object(testobject)
        path = self.store._get_shafile_path(testobject.id)
        mode = os.stat(path).st_mode
        packmode = '0o100444' if sys.platform != 'win32' else '0o100666'
        self.assertEqual(oct(mode), packmode)

    def test_corrupted_object_raise_exception(self):
        """Corrupted sha1 disk file should raise specific exception."""
        self.store.add_object(testobject)
        self.assertEqual((Blob.type_num, b'yummy data'), self.store.get_raw(testobject.id))
        self.assertTrue(self.store.contains_loose(testobject.id))
        self.assertIsNotNone(self.store._get_loose_object(testobject.id))
        path = self.store._get_shafile_path(testobject.id)
        old_mode = os.stat(path).st_mode
        os.chmod(path, 384)
        with open(path, 'wb') as f:
            f.write(b'')
        os.chmod(path, old_mode)
        expected_error_msg = 'Corrupted empty file detected'
        try:
            self.store.contains_loose(testobject.id)
        except EmptyFileException as e:
            self.assertEqual(str(e), expected_error_msg)
        try:
            self.store._get_loose_object(testobject.id)
        except EmptyFileException as e:
            self.assertEqual(str(e), expected_error_msg)
        self.assertEqual([testobject.id], list(self.store._iter_loose_objects()))

    def test_tempfile_in_loose_store(self):
        self.store.add_object(testobject)
        self.assertEqual([testobject.id], list(self.store._iter_loose_objects()))
        for i in range(256):
            dirname = os.path.join(self.store_dir, '%02x' % i)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            fd, n = tempfile.mkstemp(prefix='tmp_obj_', dir=dirname)
            os.close(fd)
        self.assertEqual([testobject.id], list(self.store._iter_loose_objects()))

    def test_add_alternate_path(self):
        store = DiskObjectStore(self.store_dir)
        self.assertEqual([], list(store._read_alternate_paths()))
        store.add_alternate_path('/foo/path')
        self.assertEqual(['/foo/path'], list(store._read_alternate_paths()))
        store.add_alternate_path('/bar/path')
        self.assertEqual(['/foo/path', '/bar/path'], list(store._read_alternate_paths()))

    def test_rel_alternative_path(self):
        alternate_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, alternate_dir)
        alternate_store = DiskObjectStore(alternate_dir)
        b2 = make_object(Blob, data=b'yummy data')
        alternate_store.add_object(b2)
        store = DiskObjectStore(self.store_dir)
        self.assertRaises(KeyError, store.__getitem__, b2.id)
        store.add_alternate_path(os.path.relpath(alternate_dir, self.store_dir))
        self.assertEqual(list(alternate_store), list(store.alternates[0]))
        self.assertIn(b2.id, store)
        self.assertEqual(b2, store[b2.id])

    def test_pack_dir(self):
        o = DiskObjectStore(self.store_dir)
        self.assertEqual(os.path.join(self.store_dir, 'pack'), o.pack_dir)

    def test_add_pack(self):
        o = DiskObjectStore(self.store_dir)
        self.addCleanup(o.close)
        f, commit, abort = o.add_pack()
        try:
            b = make_object(Blob, data=b'more yummy data')
            write_pack_objects(f.write, [(b, None)])
        except BaseException:
            abort()
            raise
        else:
            commit()

    def test_add_thin_pack(self):
        o = DiskObjectStore(self.store_dir)
        try:
            blob = make_object(Blob, data=b'yummy data')
            o.add_object(blob)
            f = BytesIO()
            entries = build_pack(f, [(REF_DELTA, (blob.id, b'more yummy data'))], store=o)
            with o.add_thin_pack(f.read, None) as pack:
                packed_blob_sha = sha_to_hex(entries[0][3])
                pack.check_length_and_checksum()
                self.assertEqual(sorted([blob.id, packed_blob_sha]), list(pack))
                self.assertTrue(o.contains_packed(packed_blob_sha))
                self.assertTrue(o.contains_packed(blob.id))
                self.assertEqual((Blob.type_num, b'more yummy data'), o.get_raw(packed_blob_sha))
        finally:
            o.close()

    def test_add_thin_pack_empty(self):
        with closing(DiskObjectStore(self.store_dir)) as o:
            f = BytesIO()
            entries = build_pack(f, [], store=o)
            self.assertEqual([], entries)
            o.add_thin_pack(f.read, None)