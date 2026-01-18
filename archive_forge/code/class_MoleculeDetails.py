import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
class MoleculeDetails(object):
    __slots__ = ['detailFP', 'scaffoldFP', 'bitInfoDetailFP', 'bitInfoScaffoldFP', 'reactivity', 'bitReactivity', 'molecule']

    def _atomDetailInvariant(self, mol):
        mol.UpdatePropertyCache(False)
        num_atoms = mol.GetNumAtoms()
        Chem.GetSSSR(mol)
        rinfo = mol.GetRingInfo()
        invariants = [0] * num_atoms
        for i, a in enumerate(mol.GetAtoms()):
            descriptors = []
            descriptors.append(a.GetAtomicNum())
            descriptors.append(a.GetTotalDegree())
            descriptors.append(a.GetTotalNumHs())
            descriptors.append(rinfo.IsAtomInRingOfSize(a.GetIdx(), 6))
            descriptors.append(rinfo.IsAtomInRingOfSize(a.GetIdx(), 5))
            descriptors.append(a.IsInRing())
            descriptors.append(a.GetIsAromatic())
            invariants[i] = hash(tuple(descriptors)) & 4294967295
        return invariants

    def _atomScaffoldInvariant(self, mol):
        num_atoms = mol.GetNumAtoms()
        invariants = [0] * num_atoms
        for i, a in enumerate(mol.GetAtoms()):
            descriptors = []
            descriptors.append(a.GetAtomicNum())
            invariants[i] = hash(tuple(descriptors)) & 4294967295
        return invariants

    def _createFP(self, mol, invariant, bitinfo, useBondTypes=True, radius=1):
        return AllChem.GetMorganFingerprint(mol=mol, radius=radius, invariants=invariant, useBondTypes=useBondTypes, bitInfo=bitinfo)

    def _isHeteroAtom(self, a):
        return a.GetAtomicNum() not in (6, 1)

    def _isSp3OrAromaticCarbon(self, a):
        if a.GetAtomicNum() != 6:
            return False
        if a.GetIsAromatic():
            return True
        for b in a.GetBonds():
            if b.GetBondTypeAsDouble() > 1.5:
                return False
        return True

    def _calcReactivityAtom(self, a):
        if self._isSp3OrAromaticCarbon(a) or (len(a.GetNeighbors()) == 0 and a.GetFormalCharge() == 0):
            return 0
        reactivity = 1
        b = a.GetBonds()
        if self._isHeteroAtom(a) or a.GetTotalNumHs() > 0:
            reactivity += 1
        if a.IsInRing():
            if a.GetIsAromatic():
                reactivity += 0.5
        else:
            reactivity += 1
        if a.GetFormalCharge():
            reactivity += 2
        for bo in b:
            ni = bo.GetOtherAtom(a)
            if bo.GetBondTypeAsDouble() > 1.5:
                reactivity += 1
                if ni.GetTotalNumHs() > 0:
                    reactivity += 1
            if self._isHeteroAtom(ni):
                reactivity += 1
                if a.GetAtomicNum() in (7, 8) and ni.GetAtomicNum() in (7, 8):
                    reactivity += 2
                elif ni.GetAtomicNum() in (12, 14, 15, 46, 50):
                    reactivity += 1
        return reactivity

    def _calcReactivityMolecule(self, mol):
        reactivityAtoms = [self._calcReactivityAtom(a) for a in mol.GetAtoms()]
        return reactivityAtoms

    def __init__(self, molecule, verbose=0):
        self.molecule = molecule
        self.bitInfoDetailFP = {}
        self.detailFP = self._createFP(molecule, self._atomDetailInvariant(molecule), self.bitInfoDetailFP)
        self.bitInfoScaffoldFP = {}
        self.scaffoldFP = self._createFP(molecule, self._atomScaffoldInvariant(molecule), self.bitInfoScaffoldFP, useBondTypes=False)
        reactivityAtoms = self._calcReactivityMolecule(molecule)
        reactivity = sum(reactivityAtoms)
        if Chem.MolToSmiles(molecule) in frequentReagents:
            reactivity *= 0.8
        self.reactivity = reactivity