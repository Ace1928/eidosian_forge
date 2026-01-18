import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def FingerprintsFromPickles(dataSource, idCol, pklCol, fingerprinter=Chem.RDKFingerprint, reportFreq=10, maxMols=-1, **fpArgs):
    """ fpArgs are passed as keyword arguments to the fingerprinter

  Returns a list of 2-tuples: (ID,fp)

  """
    res = []
    nDone = 0
    for entry in dataSource:
        ID, pkl = (str(entry[idCol]), str(entry[pklCol]))
        mol = Chem.Mol(pkl)
        if mol is not None:
            fp = FingerprintMol(mol, fingerprinter, **fpArgs)
            res.append((ID, fp))
            nDone += 1
            if reportFreq > 0 and (not nDone % reportFreq):
                message(f'Done {nDone} molecules\n')
            if maxMols > 0 and nDone >= maxMols:
                break
        else:
            error(f'Problems parsing pickle for ID: {ID}\n')
    return res