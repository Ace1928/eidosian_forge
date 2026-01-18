import os
import tempfile
from win32com.client import Dispatch
from rdkit import Chem
def ShowOnly(self, recurse=True):
    self.doc.DoCommand('HideAll')
    self.Select(state=True, recurse=True)
    self.doc.DoCommand('Show')
    self.Select(state=False, recurse=True)