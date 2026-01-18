import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def FoldFingerprintToTargetDensity(fp, **fpArgs):
    nOn = fp.GetNumOnBits()
    nTot = fp.GetNumBits()
    while float(nOn) / nTot < fpArgs['tgtDensity']:
        if nTot / 2 > fpArgs['minSize']:
            fp = DataStructs.FoldFingerprint(fp, 2)
            nOn = fp.GetNumOnBits()
            nTot = fp.GetNumBits()
        else:
            break
    return fp