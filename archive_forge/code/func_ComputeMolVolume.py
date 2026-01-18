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
def ComputeMolVolume(mol, confId=-1, gridSpacing=0.2, boxMargin=2.0):
    """ Calculates the volume of a particular conformer of a molecule
    based on a grid-encoding of the molecular shape.

    A bit of demo as well as a test of github #1883:

    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.AddHs(Chem.MolFromSmiles('C'))
    >>> AllChem.EmbedMolecule(mol)
    0
    >>> ComputeMolVolume(mol)
    28...
    >>> mol = Chem.AddHs(Chem.MolFromSmiles('O'))
    >>> AllChem.EmbedMolecule(mol)
    0
    >>> ComputeMolVolume(mol)
    20...

    """
    mol = rdchem.Mol(mol)
    conf = mol.GetConformer(confId)
    CanonicalizeConformer(conf, ignoreHs=False)
    box = ComputeConfBox(conf)
    sideLen = (box[1].x - box[0].x + 2 * boxMargin, box[1].y - box[0].y + 2 * boxMargin, box[1].z - box[0].z + 2 * boxMargin)
    shape = rdGeometry.UniformGrid3D(sideLen[0], sideLen[1], sideLen[2], spacing=gridSpacing)
    EncodeShape(mol, shape, confId, ignoreHs=False, vdwScale=1.0)
    voxelVol = gridSpacing ** 3
    occVect = shape.GetOccupancyVect()
    voxels = [1 for x in occVect if x == 3]
    vol = voxelVol * len(voxels)
    return vol