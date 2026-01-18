import numpy
from rdkit import Chem
def EStateIndices(mol, force=True):
    """ returns a tuple of EState indices for the molecule

    Reference: Hall, Mohney and Kier. JCICS _31_ 76-81 (1991)

  """
    if not force and hasattr(mol, '_eStateIndices'):
        return mol._eStateIndices
    tbl = Chem.GetPeriodicTable()
    nAtoms = mol.GetNumAtoms()
    Is = numpy.zeros(nAtoms, dtype=numpy.float64)
    for i in range(nAtoms):
        at = mol.GetAtomWithIdx(i)
        d = at.GetDegree()
        if d > 0:
            atNum = at.GetAtomicNum()
            dv = tbl.GetNOuterElecs(atNum) - at.GetTotalNumHs()
            N = GetPrincipleQuantumNumber(atNum)
            Is[i] = (4.0 / (N * N) * dv + 1) / d
    dists = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0) + 1
    accum = numpy.zeros(nAtoms, dtype=numpy.float64)
    for i in range(nAtoms):
        for j in range(i + 1, nAtoms):
            p = dists[i, j]
            if p < 1000000.0:
                tmp = (Is[i] - Is[j]) / (p * p)
                accum[i] += tmp
                accum[j] -= tmp
    res = accum + Is
    mol._eStateIndices = res
    return res