import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
class FingerprinterDetails(object):
    """ class for storing the details of a fingerprinting run,
     generates sensible defaults on construction

  """

    def __init__(self):
        self._fingerprinterInit()
        self._screenerInit()
        self._clusterInit()

    def _fingerprinterInit(self):
        self.fingerprinter = Chem.RDKFingerprint
        self.fpColName = 'AutoFragmentFP'
        self.idName = ''
        self.dbName = ''
        self.outDbName = ''
        self.tableName = ''
        self.minSize = 64
        self.fpSize = 2048
        self.tgtDensity = 0.3
        self.minPath = 1
        self.maxPath = 7
        self.discrimHash = 0
        self.useHs = 0
        self.useValence = 0
        self.bitsPerHash = 2
        self.smilesName = 'SMILES'
        self.maxMols = -1
        self.outFileName = ''
        self.outTableName = ''
        self.inFileName = ''
        self.replaceTable = True
        self.molPklName = ''
        self.useSmiles = True
        self.useSD = False

    def _screenerInit(self):
        self.metric = DataStructs.TanimotoSimilarity
        self.doScreen = ''
        self.topN = 10
        self.screenThresh = 0.75
        self.doThreshold = 0
        self.smilesTableName = ''
        self.probeSmiles = ''
        self.probeMol = None
        self.noPickle = 0

    def _clusterInit(self):
        self.clusterAlgo = Murtagh.WARDS
        self.actTableName = ''
        self.actName = ''

    def GetMetricName(self):
        metricDict = {DataStructs.DiceSimilarity: 'Dice', DataStructs.TanimotoSimilarity: 'Tanimoto', DataStructs.CosineSimilarity: 'Cosine'}
        metric = metricDict.get(self.metric, self.metric)
        if metric:
            return metric
        return 'Unknown'

    def SetMetricFromName(self, name):
        metricDict = {'DICE': DataStructs.DiceSimilarity, 'TANIMOTO': DataStructs.TanimotoSimilarity, 'COSINE': DataStructs.CosineSimilarity}
        self.metric = metricDict.get(name.upper(), self.metric)