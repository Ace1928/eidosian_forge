import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
class TestMove(BaseGroup):
    """
        Feature: Group.move moves links in a file
    """

    def test_move_hardlink(self):
        """ Moving an object """
        grp = self.f.create_group('X')
        self.f.move('X', 'Y')
        self.assertEqual(self.f['Y'], grp)
        self.f.move('Y', 'new/nested/path')
        self.assertEqual(self.f['new/nested/path'], grp)

    def test_move_softlink(self):
        """ Moving a soft link """
        self.f['soft'] = h5py.SoftLink('relative/path')
        self.f.move('soft', 'new_soft')
        lnk = self.f.get('new_soft', getlink=True)
        self.assertEqual(lnk.path, 'relative/path')

    def test_move_conflict(self):
        """ Move conflict raises ValueError """
        self.f.create_group('X')
        self.f.create_group('Y')
        with self.assertRaises(ValueError):
            self.f.move('X', 'Y')

    def test_short_circuit(self):
        """ Test that a null-move works """
        self.f.create_group('X')
        self.f.move('X', 'X')