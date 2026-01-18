import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
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