from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
class ParseReftupleTests(TestCase):

    def test_nonexistent(self):
        r = {}
        self.assertRaises(KeyError, parse_reftuple, r, r, b'thisdoesnotexist')

    def test_head(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual((b'refs/heads/foo', b'refs/heads/foo', False), parse_reftuple(r, r, b'foo'))
        self.assertEqual((b'refs/heads/foo', b'refs/heads/foo', True), parse_reftuple(r, r, b'+foo'))
        self.assertEqual((b'refs/heads/foo', b'refs/heads/foo', True), parse_reftuple(r, {}, b'+foo'))
        self.assertEqual((b'refs/heads/foo', b'refs/heads/foo', True), parse_reftuple(r, {}, b'foo', True))

    def test_full(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual((b'refs/heads/foo', b'refs/heads/foo', False), parse_reftuple(r, r, b'refs/heads/foo'))

    def test_no_left_ref(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual((None, b'refs/heads/foo', False), parse_reftuple(r, r, b':refs/heads/foo'))

    def test_no_right_ref(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual((b'refs/heads/foo', None, False), parse_reftuple(r, r, b'refs/heads/foo:'))

    def test_default_with_string(self):
        r = {b'refs/heads/foo': 'bla'}
        self.assertEqual((b'refs/heads/foo', b'refs/heads/foo', False), parse_reftuple(r, r, 'foo'))