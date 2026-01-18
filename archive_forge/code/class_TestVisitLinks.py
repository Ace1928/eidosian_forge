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
class TestVisitLinks(TestCase):
    """
        Feature: The .visit_links and .visititems_links methods allow iterative access to
        links contained in the group and its subgroups.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = ['grp1', 'grp1/grp11', 'grp1/grp12', 'grp2', 'grp2/grp21', 'grp2/grp21/grp211']
        self.links = ['linkto_grp1', 'grp1/linkto_grp11', 'grp1/linkto_grp12', 'linkto_grp2', 'grp2/linkto_grp21', 'grp2/grp21/linkto_grp211']
        for g, l in zip(self.groups, self.links):
            self.f.create_group(g)
            self.f[l] = SoftLink(f'/{g}')

    def tearDown(self):
        self.f.close()

    def test_visit_links(self):
        """ All subgroups and links are visited """
        l = []
        self.f.visit_links(l.append)
        self.assertSameElements(l, self.groups + self.links)

    def test_visititems(self):
        """ All links are visited """
        l = []
        comp = [(x, type(self.f.get(x, getlink=True))) for x in self.groups + self.links]
        self.f.visititems_links(lambda x, y: l.append((x, type(y))))
        self.assertSameElements(comp, l)

    def test_bailout(self):
        """ Returning a non-None value immediately aborts iteration """
        x = self.f.visit_links(lambda x: x)
        self.assertEqual(x, self.groups[0])
        x = self.f.visititems_links(lambda x, y: (x, type(y)))
        self.assertEqual(x, (self.groups[0], type(self.f.get(self.groups[0], getlink=True))))