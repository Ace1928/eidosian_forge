import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def GetMolLayers(original_molecule: Chem.rdchem.Mol, data_field_names: Optional[Iterable]=None, escape: Optional[str]=None, cxflag=DEFAULT_CXFLAG, enable_tautomer_hash_v2=False) -> set(HashLayer):
    """
    Generate layers of data about that could be used to identify a molecule

    :param original_molecule: molecule to obtain canonicalization layers from
    :param data_field_names: optional sequence of names of SGroup DAT fields which
       will be included in the hash.
    :param escape: optional field which can contain arbitrary information
    :param enable_tautomer_hash_v2: use v2 of the tautomer hash
    :return: dictionary of HashLayer enum to calculated hash
    """
    mol = _RemoveUnnecessaryHs(original_molecule, preserve_stereogenic_hs=True)
    _StripAtomMapLabels(mol)
    Chem.CanonicalizeEnhancedStereo(mol)
    formula = rdMolHash.MolHash(mol, rdMolHash.HashFunction.MolFormula)
    ps = Chem.SmilesWriteParams()
    cxsmiles = Chem.MolToCXSmiles(mol, ps, cxflag)
    tautomer_hash = GetStereoTautomerHash(mol, cxflag=cxflag, enable_tautomer_hash_v2=enable_tautomer_hash_v2)
    sgroup_data = _CanonicalizeSGroups(mol, dataFieldNames=data_field_names)
    no_stereo_tautomer_hash, no_stereo_smiles = GetNoStereoLayers(mol, enable_tautomer_hash_v2=enable_tautomer_hash_v2)
    return {HashLayer.CANONICAL_SMILES: cxsmiles, HashLayer.ESCAPE: escape or '', HashLayer.FORMULA: formula, HashLayer.NO_STEREO_SMILES: no_stereo_smiles, HashLayer.NO_STEREO_TAUTOMER_HASH: no_stereo_tautomer_hash, HashLayer.SGROUP_DATA: sgroup_data, HashLayer.TAUTOMER_HASH: tautomer_hash}