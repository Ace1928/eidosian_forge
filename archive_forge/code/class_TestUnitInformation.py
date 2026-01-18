import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
class TestUnitInformation(TestCase):

    def test_si(self):
        pq.set_default_units(information='B')
        self.assertQuantityEqual(pq.kB.simplified, pq.B * pq.kilo)
        self.assertQuantityEqual(pq.MB.simplified, pq.B * pq.mega)
        self.assertQuantityEqual(pq.GB.simplified, pq.B * pq.giga)
        self.assertQuantityEqual(pq.TB.simplified, pq.B * pq.tera)
        self.assertQuantityEqual(pq.PB.simplified, pq.B * pq.peta)
        self.assertQuantityEqual(pq.EB.simplified, pq.B * pq.exa)
        self.assertQuantityEqual(pq.ZB.simplified, pq.B * pq.zetta)
        self.assertQuantityEqual(pq.YB.simplified, pq.B * pq.yotta)

    def test_si_aliases(self):
        prefixes = ['kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'yotta']
        for prefix in prefixes:
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'bytes'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octet'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octets'))

    def test_iec(self):
        pq.set_default_units(information='B')
        self.assertQuantityEqual(pq.KiB.simplified, pq.B * pq.kibi)
        self.assertQuantityEqual(pq.MiB.simplified, pq.B * pq.mebi)
        self.assertQuantityEqual(pq.GiB.simplified, pq.B * pq.gibi)
        self.assertQuantityEqual(pq.TiB.simplified, pq.B * pq.tebi)
        self.assertQuantityEqual(pq.PiB.simplified, pq.B * pq.pebi)
        self.assertQuantityEqual(pq.EiB.simplified, pq.B * pq.exbi)
        self.assertQuantityEqual(pq.ZiB.simplified, pq.B * pq.zebi)
        self.assertQuantityEqual(pq.YiB.simplified, pq.B * pq.yobi)

    def test_iec_aliases(self):
        prefixes = ['kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi', 'zebi', 'yobi']
        for prefix in prefixes:
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'bytes'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octet'))
            self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octets'))