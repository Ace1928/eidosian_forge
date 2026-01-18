from unittest import TestCase
import patiencediff
from .. import multiparent, tests
class TestVersionedFile(TestCase):

    def add_version(self, vf, text, version_id, parent_ids):
        vf.add_version([bytes([t]) + b'\n' for t in bytearray(text)], version_id, parent_ids)

    def make_vf(self):
        vf = multiparent.MultiMemoryVersionedFile()
        self.add_version(vf, b'abcd', b'rev-a', [])
        self.add_version(vf, b'acde', b'rev-b', [])
        self.add_version(vf, b'abef', b'rev-c', [b'rev-a', b'rev-b'])
        return vf

    def test_add_version(self):
        vf = self.make_vf()
        self.assertEqual(REV_A, vf._lines[b'rev-a'])
        vf.clear_cache()
        self.assertEqual(vf._lines, {})

    def test_get_line_list(self):
        vf = self.make_vf()
        vf.clear_cache()
        self.assertEqual(REV_A, vf.get_line_list([b'rev-a'])[0])
        self.assertEqual([REV_B, REV_C], vf.get_line_list([b'rev-b', b'rev-c']))

    def test_reconstruct_empty(self):
        vf = multiparent.MultiMemoryVersionedFile()
        vf.add_version([], b'a', [])
        self.assertEqual([], self.reconstruct_version(vf, b'a'))

    @staticmethod
    def reconstruct(vf, revision_id, start, end):
        reconstructor = multiparent._Reconstructor(vf, vf._lines, vf._parents)
        lines = []
        reconstructor._reconstruct(lines, revision_id, start, end)
        return lines

    @staticmethod
    def reconstruct_version(vf, revision_id):
        reconstructor = multiparent._Reconstructor(vf, vf._lines, vf._parents)
        lines = []
        reconstructor.reconstruct_version(lines, revision_id)
        return lines

    def test_reconstructor(self):
        vf = self.make_vf()
        self.assertEqual([b'a\n', b'b\n'], self.reconstruct(vf, b'rev-a', 0, 2))
        self.assertEqual([b'c\n', b'd\n'], self.reconstruct(vf, b'rev-a', 2, 4))
        self.assertEqual([b'e\n', b'f\n'], self.reconstruct(vf, b'rev-c', 2, 4))
        self.assertEqual([b'a\n', b'b\n', b'e\n', b'f\n'], self.reconstruct(vf, b'rev-c', 0, 4))
        self.assertEqual([b'a\n', b'b\n', b'e\n', b'f\n'], self.reconstruct_version(vf, b'rev-c'))

    def test_reordered(self):
        """Check for a corner case that requires re-starting the cursor"""
        vf = multiparent.MultiMemoryVersionedFile()
        self.add_version(vf, b'c', b'rev-a', [])
        self.add_version(vf, b'acb', b'rev-b', [b'rev-a'])
        self.add_version(vf, b'b', b'rev-c', [b'rev-b'])
        self.add_version(vf, b'a', b'rev-d', [b'rev-b'])
        self.add_version(vf, b'ba', b'rev-e', [b'rev-c', b'rev-d'])
        vf.clear_cache()
        lines = vf.get_line_list([b'rev-e'])[0]
        self.assertEqual([b'b\n', b'a\n'], lines)