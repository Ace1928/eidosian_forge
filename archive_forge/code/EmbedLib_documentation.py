import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
 Computes the chiral volume of an atom

  We're using the chiral volume formula from Figure 7 of
  Blaney and Dixon, Rev. Comp. Chem. V, 299-335 (1994)

    >>> import os.path
    >>> from rdkit import RDConfig
    >>> dataDir = os.path.join(RDConfig.RDCodeDir,'Chem/Pharm3D/test_data')

    R configuration atoms give negative volumes:

    >>> mol = Chem.MolFromMolFile(os.path.join(dataDir, 'mol-r.mol'))
    >>> Chem.AssignStereochemistry(mol)
    >>> mol.GetAtomWithIdx(1).GetProp('_CIPCode')
    'R'
    >>> ComputeChiralVolume(mol, 1) < 0
    True

    S configuration atoms give positive volumes:

    >>> mol = Chem.MolFromMolFile(os.path.join(dataDir, 'mol-s.mol'))
    >>> Chem.AssignStereochemistry(mol)
    >>> mol.GetAtomWithIdx(1).GetProp('_CIPCode')
    'S'
    >>> ComputeChiralVolume(mol, 1) > 0
    True

    Non-chiral (or non-specified) atoms give zero volume:

    >>> ComputeChiralVolume(mol, 0) == 0.0
    True

    We also work on 3-coordinate atoms (with implicit Hs):

    >>> mol = Chem.MolFromMolFile(os.path.join(dataDir, 'mol-r-3.mol'))
    >>> Chem.AssignStereochemistry(mol)
    >>> mol.GetAtomWithIdx(1).GetProp('_CIPCode')
    'R'
    >>> ComputeChiralVolume(mol, 1) < 0
    True

    >>> mol = Chem.MolFromMolFile(os.path.join(dataDir, 'mol-s-3.mol'))
    >>> Chem.AssignStereochemistry(mol)
    >>> mol.GetAtomWithIdx(1).GetProp('_CIPCode')
    'S'
    >>> ComputeChiralVolume(mol, 1) > 0
    True


  