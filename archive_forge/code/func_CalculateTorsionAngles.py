import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def CalculateTorsionAngles(mol, tors_list, tors_list_rings, confId=-1):
    """ Calculate the torsion angles for a list of non-ring and 
      a list of ring torsions.

      Arguments:
      - mol:       the molecule of interest
      - tors_list: list of non-ring torsions
      - tors_list_rings: list of ring torsions
      - confId:    index of the conformation (default: first conformer)

      Return: list of torsion angles
  """
    torsions = []
    conf = mol.GetConformer(confId)
    for quartets, maxdev in tors_list:
        tors = []
        for atoms in quartets:
            p1, p2, p3, p4 = _getTorsionAtomPositions(atoms, conf)
            tmpTors = Geometry.ComputeSignedDihedralAngle(p1, p2, p3, p4) / math.pi * 180.0
            if tmpTors < 0:
                tmpTors += 360.0
            tors.append(tmpTors)
        torsions.append((tors, maxdev))
    for quartets, maxdev in tors_list_rings:
        num = len(quartets)
        tors = 0
        for atoms in quartets:
            p1, p2, p3, p4 = _getTorsionAtomPositions(atoms, conf)
            tmpTors = abs(Geometry.ComputeSignedDihedralAngle(p1, p2, p3, p4) / math.pi * 180.0)
            tors += tmpTors
        tors /= num
        torsions.append(([tors], maxdev))
    return torsions