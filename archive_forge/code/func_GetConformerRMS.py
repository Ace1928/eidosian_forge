import sys
import warnings
from collections import namedtuple
import numpy
from rdkit import DataStructs, ForceField, RDConfig, rdBase
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdFingerprintGenerator import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Geometry import rdGeometry
from rdkit.RDLogger import logger
def GetConformerRMS(mol, confId1, confId2, atomIds=None, prealigned=False):
    """ Returns the RMS between two conformations.
    By default, the conformers will be aligned to the first conformer
    before the RMS calculation and, as a side-effect, the second will be left
    in the aligned state.

    Arguments:
      - mol:        the molecule
      - confId1:    the id of the first conformer
      - confId2:    the id of the second conformer
      - atomIds:    (optional) list of atom ids to use a points for
                    alingment - defaults to all atoms
      - prealigned: (optional) by default the conformers are assumed
                    be unaligned and the second conformer be aligned
                    to the first

    """
    if not prealigned:
        if atomIds:
            AlignMolConformers(mol, confIds=[confId1, confId2], atomIds=atomIds)
        else:
            AlignMolConformers(mol, confIds=[confId1, confId2])
    conf1 = mol.GetConformer(id=confId1)
    conf2 = mol.GetConformer(id=confId2)
    ssr = 0
    for i in range(mol.GetNumAtoms()):
        d = conf1.GetAtomPosition(i).Distance(conf2.GetAtomPosition(i))
        ssr += d * d
    ssr /= mol.GetNumAtoms()
    return numpy.sqrt(ssr)