from rdkit import Chem
from rdkit.Chem.Pharm2D import Utils
def GetAtomsMatchingBit(sigFactory, bitIdx, mol, dMat=None, justOne=0, matchingAtoms=None):
    """ Returns a list of lists of atom indices for a bit

    **Arguments**

      - sigFactory: a SigFactory

      - bitIdx: the bit to be queried

      - mol: the molecule to be examined

      - dMat: (optional) the distance matrix of the molecule

      - justOne: (optional) if this is nonzero, only the first match
        will be returned.

      - matchingAtoms: (optional) if this is nonzero, it should
        contain a sequence of sequences with the indices of atoms in
        the molecule which match each of the patterns used by the
        signature.

    **Returns**

      a list of tuples with the matching atoms
  """
    assert sigFactory.shortestPathsOnly, 'not implemented for non-shortest path signatures'
    nPts, featCombo, scaffold = sigFactory.GetBitInfo(bitIdx)
    if _verbose:
        print('info:', nPts)
        print('\t', featCombo)
        print('\t', scaffold)
    if matchingAtoms is None:
        matchingAtoms = sigFactory.GetMolFeats(mol)
    choices = []
    for featIdx in featCombo:
        tmp = matchingAtoms[featIdx]
        if tmp:
            choices.append(tmp)
        else:
            if _verbose:
                print('no match found for feature:', featIdx)
            return []
    if _verbose:
        print('choices:')
        print(choices)
    if dMat is None:
        dMat = Chem.GetDistanceMatrix(mol, sigFactory.includeBondOrder)
    distsToCheck = Utils.nPointDistDict[nPts]
    protoPharmacophores = Utils.GetAllCombinations(choices, noDups=1)
    res = []
    for protoPharm in protoPharmacophores:
        if _verbose:
            print('protoPharm:', protoPharm)
        for i in range(len(distsToCheck)):
            dLow, dHigh = sigFactory.GetBins()[scaffold[i]]
            a1, a2 = distsToCheck[i]
            idx1, idx2 = (protoPharm[a1][0], protoPharm[a2][0])
            dist = dMat[idx1, idx2]
            if _verbose:
                print(f'\t dist: {idx1}->{idx2} = {dist} ({dLow}, {dHigh})')
            if dist < dLow or dist >= dHigh:
                break
        else:
            if _verbose:
                print('Found one')
            protoPharm.sort()
            protoPharm = tuple(protoPharm)
            if protoPharm not in res:
                res.append(protoPharm)
                if justOne:
                    break
    return res