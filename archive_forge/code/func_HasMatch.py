import sys
from rdkit import Chem
from rdkit.Chem.rdfiltercatalog import *
def HasMatch(self, mol):
    """Return True if the filter matches the molecule"""
    raise NotImplementedError('Need to implement HasMatch(mol) in a subclass of %s', self.__class__.__name__)