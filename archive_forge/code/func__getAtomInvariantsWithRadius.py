import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _getAtomInvariantsWithRadius(mol, radius):
    """ Helper function to calculate the atom invariants for each atom 
      with a given radius

      Arguments:
      - mol:    the molecule of interest
      - radius: the radius for the Morgan fingerprint

      Return: list of atom invariants
  """
    inv = []
    for i in range(mol.GetNumAtoms()):
        info = {}
        fp = rdMolDescriptors.GetMorganFingerprint(mol, radius, fromAtoms=[i], bitInfo=info)
        for k in info.keys():
            if info[k][0][1] == radius:
                inv.append(k)
    return inv