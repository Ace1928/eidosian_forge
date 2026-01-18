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
class TestRepr(BaseGroup):
    """Opened and closed groups provide a useful __repr__ string"""

    def test_repr(self):
        """ Opened and closed groups provide a useful __repr__ string """
        g = self.f.create_group('foo')
        self.assertIsInstance(repr(g), str)
        g.id._close()
        self.assertIsInstance(repr(g), str)
        g = self.f['foo']
        self.f.close()
        self.assertIsInstance(repr(g), str)