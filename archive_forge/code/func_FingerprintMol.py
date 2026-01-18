import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def FingerprintMol(mol, fingerprinter=Chem.RDKFingerprint, **fpArgs):
    if not fpArgs:
        fpArgs = FingerprinterDetails().__dict__
    if fingerprinter != Chem.RDKFingerprint:
        fp = fingerprinter(mol, **fpArgs)
        return FoldFingerprintToTargetDensity(fp, **fpArgs)
    return fingerprinter(mol, fpArgs['minPath'], fpArgs['maxPath'], fpArgs['fpSize'], fpArgs['bitsPerHash'], fpArgs['useHs'], fpArgs['tgtDensity'], fpArgs['minSize'])