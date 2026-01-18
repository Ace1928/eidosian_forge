import os
import sys
import tempfile
from rdkit import Chem
def LoadFile(self, filename, name, showOnly=False):
    """ calls pymol's "load" command on the given filename; the loaded object
    is assigned the name "name"
    """
    if showOnly:
        self.DeleteAll()
    id = self.server.loadFile(filename, name)
    return id