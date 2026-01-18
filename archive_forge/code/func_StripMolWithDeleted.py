import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def StripMolWithDeleted(self, mol, dontRemoveEverything=False):
    """
        Strips given molecule and returns it, with the fragments which have been deleted.

        >>> remover = SaltRemover(defnData="[Cl,Br]")
        >>> len(remover.salts)
        1

        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl.Br')
        >>> res, deleted = remover.StripMolWithDeleted(mol)
        >>> Chem.MolToSmiles(res)
        'CN(C)C'
        >>> [Chem.MolToSmarts(m) for m in deleted]
        ['[Cl,Br]']

        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl')
        >>> res, deleted = remover.StripMolWithDeleted(mol)
        >>> res.GetNumAtoms()
        4
        >>> len(deleted)
        1
        >>> deleted[0].GetNumAtoms()
        1
        >>> Chem.MolToSmarts(deleted[0])
        '[Cl,Br]'

        Multiple occurrences of 'Cl' and without tuple destructuring
        
        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl.Cl')
        >>> tup = remover.StripMolWithDeleted(mol)

        >>> tup.mol.GetNumAtoms()
        4
        >>> len(tup.deleted)
        1
        >>> tup.deleted[0].GetNumAtoms()
        1
        >>> Chem.MolToSmarts(deleted[0])
        '[Cl,Br]'
        """
    return self._StripMol(mol, dontRemoveEverything)