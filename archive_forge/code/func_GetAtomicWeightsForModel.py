import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetAtomicWeightsForModel(probeMol, fpFunction, predictionFunction):
    """
    Calculates the atomic weights for the probe molecule based on
    a fingerprint function and the prediction function of a ML model.

    Parameters:
      probeMol -- the probe molecule
      fpFunction -- the fingerprint function
      predictionFunction -- the prediction function of the ML model
    """
    _DeleteFpInfoAttr(probeMol)
    baseProba = predictionFunction(fpFunction(probeMol, -1))
    weights = [baseProba - predictionFunction(fpFunction(probeMol, atomId)) for atomId in range(probeMol.GetNumAtoms())]
    _DeleteFpInfoAttr(probeMol)
    return weights