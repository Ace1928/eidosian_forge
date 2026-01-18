import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def BertzCT(mol, cutoff=100, dMat=None, forceDMat=1):
    """ A topological index meant to quantify "complexity" of molecules.

     Consists of a sum of two terms, one representing the complexity
     of the bonding, the other representing the complexity of the
     distribution of heteroatoms.

     From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981)

     "cutoff" is an integer value used to limit the computational
     expense.  A cutoff value tells the program to consider vertices
     topologically identical if their distance vectors (sets of
     distances to all other vertices) are equal out to the "cutoff"th
     nearest-neighbor.

     **NOTE**  The original implementation had the following comment:
         > this implementation treats aromatic rings as the
         > corresponding Kekule structure with alternating bonds,
         > for purposes of counting "connections".
       Upon further thought, this is the WRONG thing to do.  It
        results in the possibility of a molecule giving two different
        CT values depending on the kekulization.  For example, in the
        old implementation, these two SMILES:
           CC2=CN=C1C3=C(C(C)=C(C=N3)C)C=CC1=C2C
           CC3=CN=C2C1=NC=C(C)C(C)=C1C=CC2=C3C
        which correspond to differentk kekule forms, yield different
        values.
       The new implementation uses consistent (aromatic) bond orders
        for aromatic bonds.

       THIS MEANS THAT THIS IMPLEMENTATION IS NOT BACKWARDS COMPATIBLE.

       Any molecule containing aromatic rings will yield different
       values with this implementation.  The new behavior is the correct
       one, so we're going to live with the breakage.

     **NOTE** this barfs if the molecule contains a second (or
       nth) fragment that is one atom.

  """
    atomTypeDict = {}
    connectionDict = {}
    numAtoms = mol.GetNumAtoms()
    if forceDMat or dMat is None:
        if forceDMat:
            dMat = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0, force=1)
            mol._adjMat = dMat
        else:
            try:
                dMat = mol._adjMat
            except AttributeError:
                dMat = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0, force=1)
                mol._adjMat = dMat
    if numAtoms < 2:
        return 0
    bondDict, neighborList, vdList = _CreateBondDictEtc(mol, numAtoms)
    symmetryClasses = _AssignSymmetryClasses(mol, vdList, dMat, forceDMat, numAtoms, cutoff)
    for atomIdx in range(numAtoms):
        hingeAtomNumber = mol.GetAtomWithIdx(atomIdx).GetAtomicNum()
        atomTypeDict[hingeAtomNumber] = atomTypeDict.get(hingeAtomNumber, 0) + 1
        hingeAtomClass = symmetryClasses[atomIdx]
        numNeighbors = vdList[atomIdx]
        for i in range(numNeighbors):
            neighbor_iIdx = neighborList[atomIdx][i]
            NiClass = symmetryClasses[neighbor_iIdx]
            bond_i_order = _LookUpBondOrder(atomIdx, neighbor_iIdx, bondDict)
            if bond_i_order > 1 and neighbor_iIdx > atomIdx:
                numConnections = bond_i_order * (bond_i_order - 1) / 2
                connectionKey = (min(hingeAtomClass, NiClass), max(hingeAtomClass, NiClass))
                connectionDict[connectionKey] = connectionDict.get(connectionKey, 0) + numConnections
            for j in range(i + 1, numNeighbors):
                neighbor_jIdx = neighborList[atomIdx][j]
                NjClass = symmetryClasses[neighbor_jIdx]
                bond_j_order = _LookUpBondOrder(atomIdx, neighbor_jIdx, bondDict)
                numConnections = bond_i_order * bond_j_order
                connectionKey = (min(NiClass, NjClass), hingeAtomClass, max(NiClass, NjClass))
                connectionDict[connectionKey] = connectionDict.get(connectionKey, 0) + numConnections
    if not connectionDict:
        connectionDict = {'a': 1}
    return _CalculateEntropies(connectionDict, atomTypeDict, numAtoms)