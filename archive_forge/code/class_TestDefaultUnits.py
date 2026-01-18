import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
class TestDefaultUnits(TestCase):

    def test_default_length(self):
        pq.set_default_units(length='mm')
        self.assertQuantityEqual(pq.m.simplified, 1000 * pq.mm)
        pq.set_default_units(length='m')
        self.assertQuantityEqual(pq.m.simplified, pq.m)
        self.assertQuantityEqual(pq.mm.simplified, 0.001 * pq.m)

    def test_default_system(self):
        pq.set_default_units('cgs')
        self.assertQuantityEqual(pq.kg.simplified, 1000 * pq.g)
        self.assertQuantityEqual(pq.m.simplified, 100 * pq.cm)
        pq.set_default_units('SI')
        self.assertQuantityEqual(pq.g.simplified, 0.001 * pq.kg)
        self.assertQuantityEqual(pq.mm.simplified, 0.001 * pq.m)
        pq.set_default_units('cgs', length='mm')
        self.assertQuantityEqual(pq.kg.simplified, 1000 * pq.g)
        self.assertQuantityEqual(pq.m.simplified, 1000 * pq.mm)