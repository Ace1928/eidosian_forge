import os
import tempfile
from win32com.client import Dispatch
from rdkit import Chem
def Show(self, recurse=True):
    self.Select(state=True, recurse=True)
    self.doc.DoCommand('Show')
    self.Select(state=False, recurse=True)