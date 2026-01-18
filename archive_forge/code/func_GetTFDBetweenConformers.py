import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def GetTFDBetweenConformers(mol, confIds1, confIds2, useWeights=True, maxDev='equal', symmRadius=2, ignoreColinearBonds=True):
    """ Wrapper to calculate the TFD between two list of conformers 
      of a molecule

      Arguments:
      - mol:      the molecule of interest
      - confIds1:  first list of conformer indices
      - confIds2:  second list of conformer indices
      - useWeights: flag for using torsion weights in the TFD calculation
      - maxDev:   maximal deviation used for normalization
                  'equal': all torsions are normalized using 180.0 (default)
                  'spec':  each torsion is normalized using its specific
                           maximal deviation as given in the paper
      - symmRadius: radius used for calculating the atom invariants
                    (default: 2)
      - ignoreColinearBonds: if True (default), single bonds adjacent to
                             triple bonds are ignored
                             if False, alternative not-covalently bound
                             atoms are used to define the torsion

      Return: list of TFD values
  """
    tl, tlr = CalculateTorsionLists(mol, maxDev=maxDev, symmRadius=symmRadius, ignoreColinearBonds=ignoreColinearBonds)
    torsions1 = [CalculateTorsionAngles(mol, tl, tlr, confId=cid) for cid in confIds1]
    torsions2 = [CalculateTorsionAngles(mol, tl, tlr, confId=cid) for cid in confIds2]
    tfd = []
    if useWeights:
        weights = CalculateTorsionWeights(mol, ignoreColinearBonds=ignoreColinearBonds)
    else:
        weights = None
    for t1 in torsions1:
        for t2 in torsions2:
            tfd.append(CalculateTFD(t1, t2, weights=weights))
    return tfd