import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
class SaltRemover(object):
    defnFilename = os.path.join(RDConfig.RDDataDir, 'Salts.txt')

    def __init__(self, defnFilename=None, defnData=None, defnFormat=InputFormat.SMARTS):
        if defnFilename:
            self.defnFilename = defnFilename
        self.defnData = defnData
        self.salts = None
        self.defnFormat = defnFormat
        self._initPatterns()

    def _initPatterns(self):
        """

        >>> remover = SaltRemover()
        >>> len(remover.salts)>0
        True

        Default input format is SMARTS
        >>> remover = SaltRemover(defnData="[Cl,Br]")
        >>> len(remover.salts)
        1

        >>> remover = SaltRemover(defnData="[Na+]\\nCC(=O)O", defnFormat=InputFormat.SMILES)
        >>> len(remover.salts)
        2

        >>> from rdkit import RDLogger
        >>> RDLogger.DisableLog('rdApp.error')
        >>> remover = SaltRemover(defnData="[Cl,fail]")
        Traceback (most recent call last):
          ...
        ValueError: [Cl,fail]

        >>> RDLogger.EnableLog('rdApp.error')
        """
        if self.defnData:
            from io import StringIO
            inF = StringIO(self.defnData)
            with closing(inF):
                self.salts = []
                for line in inF:
                    if line:
                        if self.defnFormat == InputFormat.SMARTS:
                            salt = _smartsFromSmartsLine(line)
                        elif self.defnFormat == InputFormat.SMILES:
                            salt = Chem.MolFromSmiles(line)
                        else:
                            raise ValueError('Unsupported format for supplier.')
                        if salt is None:
                            raise ValueError(line)
                        self.salts.append(salt)
        elif self.defnFormat == InputFormat.SMARTS:
            self.salts = [mol for mol in _getSmartsSaltsFromFile(self.defnFilename)]
        elif self.defnFormat == InputFormat.MOL:
            self.salts = [mol for mol in SDMolSupplier(self.defnFilename)]
        elif self.defnFormat == InputFormat.SMILES:
            self.salts = [mol for mol in SmilesMolSupplier(self.defnFilename)]
        else:
            raise ValueError('Unsupported format for supplier.')

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

    def _StripMol(self, mol, dontRemoveEverything=False, sanitize=True):

        def _applyPattern(m, salt, notEverything):
            nAts = m.GetNumAtoms()
            if not nAts:
                return m
            res = m
            t = Chem.DeleteSubstructs(res, salt, True)
            if not t or (notEverything and t.GetNumAtoms() == 0):
                return res
            res = t
            while res.GetNumAtoms() and nAts > res.GetNumAtoms():
                nAts = res.GetNumAtoms()
                t = Chem.DeleteSubstructs(res, salt, True)
                if notEverything and t.GetNumAtoms() == 0:
                    break
                res = t
            return res
        StrippedMol = namedtuple('StrippedMol', ['mol', 'deleted'])
        deleted = []
        if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
            return StrippedMol(mol, deleted)
        modified = False
        natoms = mol.GetNumAtoms()
        for salt in self.salts:
            mol = _applyPattern(mol, salt, dontRemoveEverything)
            if natoms != mol.GetNumAtoms():
                natoms = mol.GetNumAtoms()
                modified = True
                deleted.append(salt)
                if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
                    break
        if sanitize and modified and (mol.GetNumAtoms() > 0):
            Chem.SanitizeMol(mol)
        return StrippedMol(mol, deleted)

    def __call__(self, mol, dontRemoveEverything=False):
        """

        >>> remover = SaltRemover(defnData="[Cl,Br]")
        >>> len(remover.salts)
        1
        >>> Chem.MolToSmarts(remover.salts[0])
        '[Cl,Br]'

        >>> mol = Chem.MolFromSmiles('CN(C)C.Cl')
        >>> res = remover(mol)
        >>> res is not None
        True
        >>> res.GetNumAtoms()
        4

        """
        return self.StripMol(mol, dontRemoveEverything=dontRemoveEverything)