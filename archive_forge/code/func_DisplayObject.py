import os
import sys
import tempfile
from rdkit import Chem
def DisplayObject(self, objName):
    self.server.do('enable %s' % objName)