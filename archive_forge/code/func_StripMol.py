import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def StripMol(self, mol, dontRemoveEverything=False, sanitize=True):
    """

        >>> remover = SaltRemover(defnData="[Cl,Br]")
        >>> len(remover.salts)
        1

        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl')
        >>> res = remover.StripMol(mol)
        >>> res is not None
        True
        >>> res.GetNumAtoms()
        4

        Notice that all salts are removed:

        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl.Cl.Br')
        >>> res = remover.StripMol(mol)
        >>> res.GetNumAtoms()
        4

        Matching (e.g. "salt-like") atoms in the molecule are unchanged:

        >>> mol = Chem.MolFromSmiles('CN(Br)Cl')
        >>> res = remover.StripMol(mol)
        >>> res.GetNumAtoms()
        4

        >>> mol = Chem.MolFromSmiles('CN(Br)Cl.Cl')
        >>> res = remover.StripMol(mol)
        >>> res.GetNumAtoms()
        4

        Charged salts are handled reasonably:

        >>> mol = Chem.MolFromSmiles('C[NH+](C)(C).[Cl-]')
        >>> res = remover.StripMol(mol)
        >>> res.GetNumAtoms()
        4


        Watch out for this case (everything removed):

        >>> remover = SaltRemover()
        >>> len(remover.salts)>1
        True
        >>> mol = Chem.MolFromSmiles('CC(=O)O.[Na]')
        >>> res = remover.StripMol(mol)
        >>> res.GetNumAtoms()
        0

        dontRemoveEverything helps with this by leaving the last salt:

        >>> res = remover.StripMol(mol,dontRemoveEverything=True)
        >>> res.GetNumAtoms()
        4

        but in cases where the last salts are the same, it can't choose
        between them, so it returns all of them:

        >>> mol = Chem.MolFromSmiles('Cl.Cl')
        >>> res = remover.StripMol(mol,dontRemoveEverything=True)
        >>> res.GetNumAtoms()
        2

        """
    strippedMol = self._StripMol(mol, dontRemoveEverything, sanitize)
    return strippedMol.mol