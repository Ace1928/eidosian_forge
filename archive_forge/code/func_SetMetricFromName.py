import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def SetMetricFromName(self, name):
    metricDict = {'DICE': DataStructs.DiceSimilarity, 'TANIMOTO': DataStructs.TanimotoSimilarity, 'COSINE': DataStructs.CosineSimilarity}
    self.metric = metricDict.get(name.upper(), self.metric)