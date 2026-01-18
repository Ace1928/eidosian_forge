import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def CalculateTFD(torsions1, torsions2, weights=None):
    """ Calculate the torsion deviation fingerprint (TFD) given two lists of
      torsion angles.

      Arguments:
      - torsions1:  torsion angles of conformation 1
      - torsions2:  torsion angles of conformation 2
      - weights:    list of torsion weights (default: None)

      Return: TFD value (float)
  """
    if len(torsions1) != len(torsions2):
        raise ValueError('List of torsions angles must have the same size.')
    deviations = []
    for tors1, tors2 in zip(torsions1, torsions2):
        mindiff = 180.0
        for t1 in tors1[0]:
            for t2 in tors2[0]:
                diff = abs(t1 - t2)
                if 360.0 - diff < diff:
                    diff = 360.0 - diff
                if diff < mindiff:
                    mindiff = diff
        deviations.append(mindiff / tors1[1])
    if weights is not None:
        if len(weights) != len(torsions1):
            raise ValueError('List of torsions angles and weights must have the same size.')
        deviations = [d * w for d, w in zip(deviations, weights)]
        sum_weights = sum(weights)
    else:
        sum_weights = len(deviations)
    tfd = sum(deviations)
    if sum_weights != 0:
        tfd /= sum_weights
    return tfd