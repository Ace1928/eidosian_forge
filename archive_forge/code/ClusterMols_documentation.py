import pickle
import numpy
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols, MolSimilarity
from rdkit.ML.Cluster import Murtagh
 Returns the cluster tree

  