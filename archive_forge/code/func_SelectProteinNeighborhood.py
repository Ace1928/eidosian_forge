import os
import sys
import tempfile
from rdkit import Chem
def SelectProteinNeighborhood(self, aroundObj, inObj, distance=5.0, name='neighborhood', showSurface=False):
    """ selects the area of a protein around a specified object/selection name;
    optionally adds a surface to that """
    self.server.do('select %(name)s,byres (%(aroundObj)s around %(distance)f) and %(inObj)s' % locals())
    if showSurface:
        self.server.do('show surface,%s' % name)
        self.server.do('disable %s' % name)