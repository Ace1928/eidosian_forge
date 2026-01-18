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
class ObjectStoreGraphWalkerTests(TestCase):

    def get_walker(self, heads, parent_map):
        new_parent_map = {k * 40: [p * 40 for p in ps] for k, ps in parent_map.items()}
        return ObjectStoreGraphWalker([x * 40 for x in heads], new_parent_map.__getitem__)

    def test_ack_invalid_value(self):
        gw = self.get_walker([], {})
        self.assertRaises(ValueError, gw.ack, 'tooshort')

    def test_empty(self):
        gw = self.get_walker([], {})
        self.assertIs(None, next(gw))
        gw.ack(b'a' * 40)
        self.assertIs(None, next(gw))

    def test_descends(self):
        gw = self.get_walker([b'a'], {b'a': [b'b'], b'b': []})
        self.assertEqual(b'a' * 40, next(gw))
        self.assertEqual(b'b' * 40, next(gw))

    def test_present(self):
        gw = self.get_walker([b'a'], {b'a': [b'b'], b'b': []})
        gw.ack(b'a' * 40)
        self.assertIs(None, next(gw))

    def test_parent_present(self):
        gw = self.get_walker([b'a'], {b'a': [b'b'], b'b': []})
        self.assertEqual(b'a' * 40, next(gw))
        gw.ack(b'a' * 40)
        self.assertIs(None, next(gw))

    def test_child_ack_later(self):
        gw = self.get_walker([b'a'], {b'a': [b'b'], b'b': [b'c'], b'c': []})
        self.assertEqual(b'a' * 40, next(gw))
        self.assertEqual(b'b' * 40, next(gw))
        gw.ack(b'a' * 40)
        self.assertIs(None, next(gw))

    def test_only_once(self):
        gw = self.get_walker([b'a', b'b'], {b'a': [b'c'], b'b': [b'd'], b'c': [b'e'], b'd': [b'e'], b'e': []})
        walk = []
        acked = False
        walk.append(next(gw))
        walk.append(next(gw))
        if walk == [b'a' * 40, b'c' * 40] or walk == [b'b' * 40, b'd' * 40]:
            gw.ack(walk[0])
            acked = True
        walk.append(next(gw))
        if not acked and walk[2] == b'c' * 40:
            gw.ack(b'a' * 40)
        elif not acked and walk[2] == b'd' * 40:
            gw.ack(b'b' * 40)
        walk.append(next(gw))
        self.assertIs(None, next(gw))
        self.assertEqual([b'a' * 40, b'b' * 40, b'c' * 40, b'd' * 40], sorted(walk))
        self.assertLess(walk.index(b'a' * 40), walk.index(b'c' * 40))
        self.assertLess(walk.index(b'b' * 40), walk.index(b'd' * 40))