import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _AssignSymmetryClasses(mol, vdList, bdMat, forceBDMat, numAtoms, cutoff):
    """
     Used by BertzCT

     vdList: the number of neighbors each atom has
     bdMat: "balaban" distance matrix

  """
    if forceBDMat:
        bdMat = Chem.GetDistanceMatrix(mol, useBO=1, useAtomWts=0, force=1, prefix='Balaban')
        mol._balabanMat = bdMat
    keysSeen = []
    symList = [0] * numAtoms
    for i in range(numAtoms):
        tmpList = bdMat[i].tolist()
        tmpList.sort()
        theKey = tuple(['%.4f' % x for x in tmpList[:cutoff]])
        try:
            idx = keysSeen.index(theKey)
        except ValueError:
            idx = len(keysSeen)
            keysSeen.append(theKey)
        symList[i] = idx + 1
    return tuple(symList)