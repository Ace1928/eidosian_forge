import pickle
import numpy
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols, MolSimilarity
from rdkit.ML.Cluster import Murtagh
def ClusterFromDetails(details):
    """ Returns the cluster tree

  """
    data = MolSimilarity.GetFingerprints(details)
    if details.maxMols > 0:
        data = data[:details.maxMols]
    if details.outFileName:
        try:
            outF = open(details.outFileName, 'wb+')
        except IOError:
            error('Error: could not open output file %s for writing\n' % details.outFileName)
            return None
    else:
        outF = None
    if not data:
        return None
    clustTree = ClusterPoints(data, details.metric, details.clusterAlgo, haveLabels=0, haveActs=1)
    if outF:
        pickle.dump(clustTree, outF)
    return clustTree