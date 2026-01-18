import os
import sys
import tempfile
from rdkit import Chem
def DeleteAll(self):
    """ blows out everything in the viewer """
    self.server.deleteAll()