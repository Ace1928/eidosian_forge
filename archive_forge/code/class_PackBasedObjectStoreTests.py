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
class PackBasedObjectStoreTests(ObjectStoreTests):

    def tearDown(self):
        for pack in self.store.packs:
            pack.close()

    def test_empty_packs(self):
        self.assertEqual([], list(self.store.packs))

    def test_pack_loose_objects(self):
        b1 = make_object(Blob, data=b'yummy data')
        self.store.add_object(b1)
        b2 = make_object(Blob, data=b'more yummy data')
        self.store.add_object(b2)
        b3 = make_object(Blob, data=b'even more yummy data')
        b4 = make_object(Blob, data=b'and more yummy data')
        self.store.add_objects([(b3, None), (b4, None)])
        self.assertEqual({b1.id, b2.id, b3.id, b4.id}, set(self.store))
        self.assertEqual(1, len(self.store.packs))
        self.assertEqual(2, self.store.pack_loose_objects())
        self.assertNotEqual([], list(self.store.packs))
        self.assertEqual(0, self.store.pack_loose_objects())

    def test_repack(self):
        b1 = make_object(Blob, data=b'yummy data')
        self.store.add_object(b1)
        b2 = make_object(Blob, data=b'more yummy data')
        self.store.add_object(b2)
        b3 = make_object(Blob, data=b'even more yummy data')
        b4 = make_object(Blob, data=b'and more yummy data')
        self.store.add_objects([(b3, None), (b4, None)])
        b5 = make_object(Blob, data=b'and more data')
        b6 = make_object(Blob, data=b'and some more data')
        self.store.add_objects([(b5, None), (b6, None)])
        self.assertEqual({b1.id, b2.id, b3.id, b4.id, b5.id, b6.id}, set(self.store))
        self.assertEqual(2, len(self.store.packs))
        self.assertEqual(6, self.store.repack())
        self.assertEqual(1, len(self.store.packs))
        self.assertEqual(0, self.store.pack_loose_objects())

    def test_repack_existing(self):
        b1 = make_object(Blob, data=b'yummy data')
        self.store.add_object(b1)
        b2 = make_object(Blob, data=b'more yummy data')
        self.store.add_object(b2)
        self.store.add_objects([(b1, None), (b2, None)])
        self.store.add_objects([(b2, None)])
        self.assertEqual({b1.id, b2.id}, set(self.store))
        self.assertEqual(2, len(self.store.packs))
        self.assertEqual(2, self.store.repack())
        self.assertEqual(1, len(self.store.packs))
        self.assertEqual(0, self.store.pack_loose_objects())
        self.assertEqual({b1.id, b2.id}, set(self.store))
        self.assertEqual(1, len(self.store.packs))
        self.assertEqual(2, self.store.repack())
        self.assertEqual(1, len(self.store.packs))
        self.assertEqual(0, self.store.pack_loose_objects())