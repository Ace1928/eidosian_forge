import argparse
import os
import re
from rdkit import Chem, RDLogger
from rdkit.Chem import ChemicalFeatures
def GetAtomFeatInfo(factory, mol):
    res = [None] * mol.GetNumAtoms()
    feats = factory.GetFeaturesForMol(mol)
    for feat in feats:
        ids = feat.GetAtomIds()
        feature = '%s-%s' % (feat.GetFamily(), feat.GetType())
        for id_ in ids:
            if res[id_] is None:
                res[id_] = []
            res[id_].append(feature)
    return res