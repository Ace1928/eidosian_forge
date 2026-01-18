import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def GetNoStereoLayers(mol, enable_tautomer_hash_v2=False):
    no_stereo_mol = _RemoveUnnecessaryHs(mol)
    no_stereo_mol.UpdatePropertyCache(False)
    Chem.rdmolops.RemoveStereochemistry(no_stereo_mol)
    if enable_tautomer_hash_v2:
        hash_func = rdMolHash.HashFunction.HetAtomTautomerv2
    else:
        hash_func = rdMolHash.HashFunction.HetAtomTautomer
    no_stereo_tautomer_hash = rdMolHash.MolHash(no_stereo_mol, hash_func)
    no_stereo_smiles = rdMolHash.MolHash(no_stereo_mol, rdMolHash.HashFunction.CanonicalSmiles)
    return (no_stereo_tautomer_hash, no_stereo_smiles)