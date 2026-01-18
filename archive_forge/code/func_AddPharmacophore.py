import os
import sys
import tempfile
from rdkit import Chem
def AddPharmacophore(self, locs, colors, label, sphereRad=0.5):
    """ adds a set of spheres """
    self.server.do('view rdinterface,store')
    self.server.resetCGO(label)
    for i, loc in enumerate(locs):
        self.server.sphere(loc, sphereRad, colors[i], label, 1)
    self.server.do('enable %s' % label)
    self.server.do('view rdinterface,recall')