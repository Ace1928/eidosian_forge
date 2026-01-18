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
class BaseMapping(BaseGroup):
    """
        Base class for mapping tests
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = ('a', 'b', 'c', 'd')
        for x in self.groups:
            self.f.create_group(x)
        self.f['x'] = h5py.SoftLink('/mongoose')
        self.groups = self.groups + ('x',)

    def tearDown(self):
        if self.f:
            self.f.close()