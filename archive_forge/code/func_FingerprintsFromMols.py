import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def FingerprintsFromMols(mols, fingerprinter=Chem.RDKFingerprint, reportFreq=10, maxMols=-1, **fpArgs):
    """ fpArgs are passed as keyword arguments to the fingerprinter

  Returns a list of 2-tuples: (ID,fp)

  """
    res = []
    nDone = 0
    for ID, mol in mols:
        if mol:
            fp = FingerprintMol(mol, fingerprinter, **fpArgs)
            res.append((ID, fp))
            nDone += 1
            if reportFreq > 0 and (not nDone % reportFreq):
                message(f'Done {nDone} molecules\n')
            if maxMols > 0 and nDone >= maxMols:
                break
        else:
            error(f'Problems parsing SMILES: {smi}\n')
    return res