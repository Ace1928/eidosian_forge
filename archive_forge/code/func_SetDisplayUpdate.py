import os
import sys
import tempfile
from rdkit import Chem
def SetDisplayUpdate(self, val):
    if not val:
        self.server.do('set defer_update,1')
    else:
        self.server.do('set defer_update,0')