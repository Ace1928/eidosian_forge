import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
class PackedRefsFileTests(TestCase):

    def test_split_ref_line_errors(self):
        self.assertRaises(errors.PackedRefsException, _split_ref_line, b'singlefield')
        self.assertRaises(errors.PackedRefsException, _split_ref_line, b'badsha name')
        self.assertRaises(errors.PackedRefsException, _split_ref_line, ONES + b' bad/../refname')

    def test_read_without_peeled(self):
        f = BytesIO(b'\n'.join([b'# comment', ONES + b' ref/1', TWOS + b' ref/2']))
        self.assertEqual([(ONES, b'ref/1'), (TWOS, b'ref/2')], list(read_packed_refs(f)))

    def test_read_without_peeled_errors(self):
        f = BytesIO(b'\n'.join([ONES + b' ref/1', b'^' + TWOS]))
        self.assertRaises(errors.PackedRefsException, list, read_packed_refs(f))

    def test_read_with_peeled(self):
        f = BytesIO(b'\n'.join([ONES + b' ref/1', TWOS + b' ref/2', b'^' + THREES, FOURS + b' ref/4']))
        self.assertEqual([(ONES, b'ref/1', None), (TWOS, b'ref/2', THREES), (FOURS, b'ref/4', None)], list(read_packed_refs_with_peeled(f)))

    def test_read_with_peeled_errors(self):
        f = BytesIO(b'\n'.join([b'^' + TWOS, ONES + b' ref/1']))
        self.assertRaises(errors.PackedRefsException, list, read_packed_refs(f))
        f = BytesIO(b'\n'.join([ONES + b' ref/1', b'^' + TWOS, b'^' + THREES]))
        self.assertRaises(errors.PackedRefsException, list, read_packed_refs(f))

    def test_write_with_peeled(self):
        f = BytesIO()
        write_packed_refs(f, {b'ref/1': ONES, b'ref/2': TWOS}, {b'ref/1': THREES})
        self.assertEqual(b'\n'.join([b'# pack-refs with: peeled', ONES + b' ref/1', b'^' + THREES, TWOS + b' ref/2']) + b'\n', f.getvalue())

    def test_write_without_peeled(self):
        f = BytesIO()
        write_packed_refs(f, {b'ref/1': ONES, b'ref/2': TWOS})
        self.assertEqual(b'\n'.join([ONES + b' ref/1', TWOS + b' ref/2']) + b'\n', f.getvalue())