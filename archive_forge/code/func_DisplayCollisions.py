import os
import sys
import tempfile
from rdkit import Chem
def DisplayCollisions(self, objName, molName, proteinName, distCutoff=3.0, color='red', molSelText='(%(molName)s)', proteinSelText='(%(proteinName)s and not het)'):
    """ toggles display of collisions between the protein and a specified molecule """
    cmd = 'delete %(objName)s;\n'
    cmd += 'dist %(objName)s,' + molSelText + ',' + proteinSelText + ',%(distCutoff)f,mode=0;\n'
    cmd += 'enable %(objName)s\n    color %(color)s, %(objName)s'
    cmd = cmd % locals()
    self.server.do(cmd)