import bisect
from rdkit import DataStructs
from rdkit.DataStructs.TopNContainer import TopNContainer
class TopNOverallPicker(GenericPicker):
    """  A class for picking the top N overall best matches across a library

  Connect to a database and build molecules:

  >>> from rdkit import Chem
  >>> from rdkit import RDConfig
  >>> import os.path
  >>> from rdkit.Dbase.DbConnection import DbConnect
  >>> dbName = RDConfig.RDTestDatabase
  >>> conn = DbConnect(dbName,'simple_mols1')
  >>> [x.upper() for x in conn.GetColumnNames()]
  ['SMILES', 'ID']
  >>> mols = []
  >>> for smi,id in conn.GetData():
  ...   mol = Chem.MolFromSmiles(str(smi))
  ...   mol.SetProp('_Name',str(id))
  ...   mols.append(mol)
  >>> len(mols)
  12

  Calculate fingerprints:

  >>> probefps = []
  >>> for mol in mols:
  ...   fp = Chem.RDKFingerprint(mol)
  ...   fp._id = mol.GetProp('_Name')
  ...   probefps.append(fp)

  Start by finding the top matches for a single probe.  This ether should pull
  other ethers from the db:

  >>> mol = Chem.MolFromSmiles('COC')
  >>> probeFp = Chem.RDKFingerprint(mol)
  >>> picker = TopNOverallPicker(numToPick=2,probeFps=[probeFp],dataSet=probefps)
  >>> len(picker)
  2
  >>> fp,score = picker[0]
  >>> id = fp._id
  >>> str(id)
  'ether-1'
  >>> score
  1.0

  The results come back in order:

  >>> fp,score = picker[1]
  >>> id = fp._id
  >>> str(id)
  'ether-2'

  Now find the top matches for 2 probes.  We'll get one ether and one acid:

  >>> fps = []
  >>> fps.append(Chem.RDKFingerprint(Chem.MolFromSmiles('COC')))
  >>> fps.append(Chem.RDKFingerprint(Chem.MolFromSmiles('CC(=O)O')))
  >>> picker = TopNOverallPicker(numToPick=3,probeFps=fps,dataSet=probefps)
  >>> len(picker)
  3
  >>> fp,score = picker[0]
  >>> id = fp._id
  >>> str(id)
  'acid-1'
  >>> fp,score = picker[1]
  >>> id = fp._id
  >>> str(id)
  'ether-1'
  >>> score
  1.0
  >>> fp,score = picker[2]
  >>> id = fp._id
  >>> str(id)
  'acid-2'

  """

    def __init__(self, numToPick=10, probeFps=None, dataSet=None, simMetric=DataStructs.TanimotoSimilarity):
        """

      dataSet should be a sequence of BitVectors

    """
        self.numToPick = numToPick
        self.probes = probeFps
        self.data = dataSet
        self.simMetric = simMetric
        self._picks = None

    def MakePicks(self, force=False):
        if self._picks is not None and (not force):
            return
        picks = TopNContainer(self.numToPick)
        for fp in self.data:
            origFp = fp
            bestScore = -1.0
            for probeFp in self.probes:
                score = DataStructs.FingerprintSimilarity(origFp, probeFp, self.simMetric)
                bestScore = max(score, bestScore)
            picks.Insert(bestScore, fp)
        self._picks = []
        for score, pt in picks:
            self._picks.append((pt, score))
        self._picks.reverse()