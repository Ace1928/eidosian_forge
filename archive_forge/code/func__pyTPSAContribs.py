import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
def _pyTPSAContribs(mol, verbose=False):
    """ DEPRECATED: this has been reimplmented in C++
  calculates atomic contributions to a molecules TPSA

   Algorithm described in:
    P. Ertl, B. Rohde, P. Selzer
     Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-based
     Contributions and Its Application to the Prediction of Drug Transport
     Properties, J.Med.Chem. 43, 3714-3717, 2000

   Implementation based on the Daylight contrib program tpsa.c

   NOTE: The JMC paper describing the TPSA algorithm includes
   contributions from sulfur and phosphorus, however according to
   Peter Ertl (personal communication, 2010) the correlation of TPSA
   with various ADME properties is better if only contributions from
   oxygen and nitrogen are used. This matches the daylight contrib
   implementation.

  """
    res = [0] * mol.GetNumAtoms()
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atNum = atom.GetAtomicNum()
        if atNum in [7, 8]:
            nHs = atom.GetTotalNumHs()
            chg = atom.GetFormalCharge()
            in3Ring = atom.IsInRingSize(3)
            bonds = atom.GetBonds()
            numNeighbors = atom.GetDegree()
            nSing = 0
            nDoub = 0
            nTrip = 0
            nArom = 0
            for bond in bonds:
                otherAt = bond.GetOtherAtom(atom)
                if otherAt.GetAtomicNum() != 1:
                    if bond.GetIsAromatic():
                        nArom += 1
                    else:
                        order = bond.GetBondType()
                        if order == Chem.BondType.SINGLE:
                            nSing += 1
                        elif order == Chem.BondType.DOUBLE:
                            nDoub += 1
                        elif order == Chem.BondType.TRIPLE:
                            nTrip += 1
                else:
                    numNeighbors -= 1
                    nHs += 1
            tmp = -1
            if atNum == 7:
                if numNeighbors == 1:
                    if nHs == 0 and nTrip == 1 and (chg == 0):
                        tmp = 23.79
                    elif nHs == 1 and nDoub == 1 and (chg == 0):
                        tmp = 23.85
                    elif nHs == 2 and nSing == 1 and (chg == 0):
                        tmp = 26.02
                    elif nHs == 2 and nDoub == 1 and (chg == 1):
                        tmp = 25.59
                    elif nHs == 3 and nSing == 1 and (chg == 1):
                        tmp = 27.64
                elif numNeighbors == 2:
                    if nHs == 0 and nSing == 1 and (nDoub == 1) and (chg == 0):
                        tmp = 12.36
                    elif nHs == 0 and nTrip == 1 and (nDoub == 1) and (chg == 0):
                        tmp = 13.6
                    elif nHs == 1 and nSing == 2 and (chg == 0):
                        if not in3Ring:
                            tmp = 12.03
                        else:
                            tmp = 21.94
                    elif nHs == 0 and nTrip == 1 and (nSing == 1) and (chg == 1):
                        tmp = 4.36
                    elif nHs == 1 and nDoub == 1 and (nSing == 1) and (chg == 1):
                        tmp = 13.97
                    elif nHs == 2 and nSing == 2 and (chg == 1):
                        tmp = 16.61
                    elif nHs == 0 and nArom == 2 and (chg == 0):
                        tmp = 12.89
                    elif nHs == 1 and nArom == 2 and (chg == 0):
                        tmp = 15.79
                    elif nHs == 1 and nArom == 2 and (chg == 1):
                        tmp = 14.14
                elif numNeighbors == 3:
                    if nHs == 0 and nSing == 3 and (chg == 0):
                        if not in3Ring:
                            tmp = 3.24
                        else:
                            tmp = 3.01
                    elif nHs == 0 and nSing == 1 and (nDoub == 2) and (chg == 0):
                        tmp = 11.68
                    elif nHs == 0 and nSing == 2 and (nDoub == 1) and (chg == 1):
                        tmp = 3.01
                    elif nHs == 1 and nSing == 3 and (chg == 1):
                        tmp = 4.44
                    elif nHs == 0 and nArom == 3 and (chg == 0):
                        tmp = 4.41
                    elif nHs == 0 and nSing == 1 and (nArom == 2) and (chg == 0):
                        tmp = 4.93
                    elif nHs == 0 and nDoub == 1 and (nArom == 2) and (chg == 0):
                        tmp = 8.39
                    elif nHs == 0 and nArom == 3 and (chg == 1):
                        tmp = 4.1
                    elif nHs == 0 and nSing == 1 and (nArom == 2) and (chg == 1):
                        tmp = 3.88
                elif numNeighbors == 4:
                    if nHs == 0 and nSing == 4 and (chg == 1):
                        tmp = 0.0
                if tmp < 0.0:
                    tmp = 30.5 - numNeighbors * 8.2 + nHs * 1.5
                    if tmp < 0.0:
                        tmp = 0.0
            elif atNum == 8:
                if numNeighbors == 1:
                    if nHs == 0 and nDoub == 1 and (chg == 0):
                        tmp = 17.07
                    elif nHs == 1 and nSing == 1 and (chg == 0):
                        tmp = 20.23
                    elif nHs == 0 and nSing == 1 and (chg == -1):
                        tmp = 23.06
                elif numNeighbors == 2:
                    if nHs == 0 and nSing == 2 and (chg == 0):
                        if not in3Ring:
                            tmp = 9.23
                        else:
                            tmp = 12.53
                    elif nHs == 0 and nArom == 2 and (chg == 0):
                        tmp = 13.14
                if tmp < 0.0:
                    tmp = 28.5 - numNeighbors * 8.6 + nHs * 1.5
                    if tmp < 0.0:
                        tmp = 0.0
            if verbose:
                print('\t', atom.GetIdx(), atom.GetSymbol(), atNum, nHs, nSing, nDoub, nTrip, nArom, chg, tmp)
            res[atom.GetIdx()] = tmp
    return res