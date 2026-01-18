import os
import sys
import tempfile
from rdkit import Chem
def SelectAtoms(self, itemId, atomIndices, selName='selection'):
    """ selects a set of atoms """
    ids = '(id '
    ids += ','.join(['%d' % (x + 1) for x in atomIndices])
    ids += ')'
    cmd = 'select %s,%s and %s' % (selName, ids, itemId)
    self.server.do(cmd)