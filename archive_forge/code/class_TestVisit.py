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
class TestVisit(TestCase):
    """
        Feature: The .visit and .visititems methods allow iterative access to
        group and subgroup members
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = ['grp1', 'grp1/sg1', 'grp1/sg2', 'grp2', 'grp2/sg1', 'grp2/sg1/ssg1']
        for x in self.groups:
            self.f.create_group(x)

    def tearDown(self):
        self.f.close()

    def test_visit(self):
        """ All subgroups are visited """
        l = []
        self.f.visit(l.append)
        self.assertSameElements(l, self.groups)

    def test_visititems(self):
        """ All subgroups and contents are visited """
        l = []
        comp = [(x, self.f[x]) for x in self.groups]
        self.f.visititems(lambda x, y: l.append((x, y)))
        self.assertSameElements(comp, l)

    def test_bailout(self):
        """ Returning a non-None value immediately aborts iteration """
        x = self.f.visit(lambda x: x)
        self.assertEqual(x, self.groups[0])
        x = self.f.visititems(lambda x, y: (x, y))
        self.assertEqual(x, (self.groups[0], self.f[self.groups[0]]))