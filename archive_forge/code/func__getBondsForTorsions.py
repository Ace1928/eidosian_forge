import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _getBondsForTorsions(mol, ignoreColinearBonds):
    """ Determine the bonds (or pair of atoms treated like a bond) for which
      torsions should be calculated.

      Arguments:
      - refmol: the molecule of interest
      - ignoreColinearBonds: if True (default), single bonds adjacent to
                             triple bonds are ignored
                             if False, alternative not-covalently bound
                             atoms are used to define the torsion
  """
    patts = [Chem.MolFromSmarts(x) for x in ['*#*', '[$([C](=*)=*)]']]
    atomFlags = [0] * mol.GetNumAtoms()
    for p in patts:
        if mol.HasSubstructMatch(p):
            matches = mol.GetSubstructMatches(p)
            for match in matches:
                for a in match:
                    atomFlags[a] = 1
    bonds = []
    doneBonds = [0] * mol.GetNumBonds()
    for b in mol.GetBonds():
        if b.IsInRing():
            continue
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        nb1 = _getHeavyAtomNeighbors(b.GetBeginAtom(), a2)
        nb2 = _getHeavyAtomNeighbors(b.GetEndAtom(), a1)
        if not doneBonds[b.GetIdx()] and (nb1 and nb2):
            doneBonds[b.GetIdx()] = 1
            if atomFlags[a1] or atomFlags[a2]:
                if not ignoreColinearBonds:
                    while len(nb1) == 1 and atomFlags[a1]:
                        a1old = a1
                        a1 = nb1[0].GetIdx()
                        b = mol.GetBondBetweenAtoms(a1old, a1)
                        if b.GetEndAtom().GetIdx() == a1old:
                            nb1 = _getHeavyAtomNeighbors(b.GetBeginAtom(), a1old)
                        else:
                            nb1 = _getHeavyAtomNeighbors(b.GetEndAtom(), a1old)
                        doneBonds[b.GetIdx()] = 1
                    while len(nb2) == 1 and atomFlags[a2]:
                        doneBonds[b.GetIdx()] = 1
                        a2old = a2
                        a2 = nb2[0].GetIdx()
                        b = mol.GetBondBetweenAtoms(a2old, a2)
                        if b.GetBeginAtom().GetIdx() == a2old:
                            nb2 = _getHeavyAtomNeighbors(b.GetEndAtom(), a2old)
                        else:
                            nb2 = _getHeavyAtomNeighbors(b.GetBeginAtom(), a2old)
                        doneBonds[b.GetIdx()] = 1
                    if nb1 and nb2:
                        bonds.append((a1, a2, nb1, nb2))
            else:
                bonds.append((a1, a2, nb1, nb2))
    return bonds