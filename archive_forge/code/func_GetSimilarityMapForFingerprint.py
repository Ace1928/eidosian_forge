import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetSimilarityMapForFingerprint(refMol, probeMol, fpFunction, metric=DataStructs.DiceSimilarity, **kwargs):
    """
    Generates the similarity map for a given reference and probe molecule,
    fingerprint function and similarity metric.

    Parameters:
      refMol -- the reference molecule
      probeMol -- the probe molecule
      fpFunction -- the fingerprint function
      metric -- the similarity metric.
      kwargs -- additional arguments for drawing
    """
    weights = GetAtomicWeightsForFingerprint(refMol, probeMol, fpFunction, metric)
    weights, maxWeight = GetStandardizedWeights(weights)
    fig = GetSimilarityMapFromWeights(probeMol, weights, **kwargs)
    return (fig, maxWeight)