import csv
import itertools
import logging
import math
import re
import sys
from collections import defaultdict, namedtuple
from typing import Generator, List
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, molzip
from rdkit.Chem import rdRGroupDecomposition as rgd
def FWDecompose(scaffolds, mols, scores, decomp_params=default_decomp_params) -> FreeWilsonDecomposition:
    """
    Perform a free wilson analysis
        : param scaffolds : scaffold or list of scaffolds to use for the rgroup decomposition
        : param mols : molecules to decompose
        : param scores : list of floating point numbers for the regression (
                             you may need convert these to their logs in some cases)
        : param decomp_params : RgroupDecompositionParams default [
                                    default_decomp_params = rdkit.Chem.rdRGroupDecomposition.RGroupDecompositionParameters()
                                    default_decomp_params.matchingStrategy = rgd.GA
                                    default_decomp_params.onlyMatchAtRGroups = False
                                   ]
                                If you only want to decompose on specific group locations
                                set onlyMatchAtRGroups to True


        >>> from rdkit import Chem
        >>> from freewilson import FWBuild, FWDecompose
        >>> from rdkit.Chem import Descriptors
        >>> scaffold = Chem.MolFromSmiles("c1cccnc1")
        >>> mols = [Chem.MolFromSmiles("c1cccnc1"+"C"*(i+1)) for i in range(100)]
        >>> scores = [Descriptors.MolLogP(m) for m in mols]
        >>> fw = FWDecompose(scaffold, mols, scores)
        >>> for pred in FWBuild(fw):
        ...   print(pred)

    For an easy way to report predictions see 

       >>> from freewilson import FWBuild, predictions_to_csv
       >>> import sys
       >>> predictions_to_csv(sys.stdout, FWBuild(fw))

   
    See FWBuild docs to see how to filter predictions, molecular weight or molecular properties.
    """
    descriptors = []
    matched_scores = []
    rgroup_idx = {}
    rgroups = defaultdict(list)
    if len(mols) != len(scores):
        raise ValueError(f'The number of molecules must match the number of scores #mols {len(mols)} #scores {len(scores)}')
    logger.info(f'Decomposing {len(mols)} molecules...')
    decomposer = rgd.RGroupDecomposition(scaffolds, decomp_params)
    matched = []
    matched_indices = []
    for i, (mol, score) in enumerate(tqdm(zip(mols, scores))):
        if decomposer.Add(mol) >= 0:
            matched_scores.append(score)
            matched.append(mol)
            matched_indices.append(i)
    decomposer.Process()
    logger.info(f'Matched {len(matched_scores)} out of {len(mols)}')
    if not matched_scores:
        logger.error('No scaffolds matched the input molecules')
        return
    decomposition = decomposer.GetRGroupsAsRows(asSmiles=True)
    logger.info('Get unique rgroups...')
    blocker = rdBase.BlockLogs()
    rgroup_counts = defaultdict(int)
    num_reconstructed = 0
    for num_mols, (row, idx) in enumerate(zip(decomposition, matched_indices)):
        row_smiles = []
        for rgroup, smiles in row.items():
            row_smiles.append(smiles)
            rgroup_counts[smiles] += 1
            if smiles not in rgroup_idx:
                rgroup_idx[smiles] = len(rgroup_idx)
                rgroups[rgroup].append(RGroup(smiles, rgroup, 0, 0))
        row['original_idx'] = idx
        reconstructed = '.'.join(row_smiles)
        try:
            blocker = rdBase.BlockLogs()
            mol = molzip_smi(reconstructed)
            num_reconstructed += 1
        except:
            print('failed:', Chem.MolToSmiles(matched[num_mols]), reconstructed)
    logger.info(f'Descriptor size {len(rgroup_idx)}')
    logger.info(f'Reconstructed {num_reconstructed} out of {num_mols}')
    if num_reconstructed == 0:
        logging.warning('Could only reconstruct %s out of %s training molecules', num_mols, num_reconstructed)
    for mol, row in zip(matched, decomposition):
        row['molecule'] = mol
        descriptor = [0] * len(rgroup_idx)
        descriptors.append(descriptor)
        for smiles in row.values():
            if smiles in rgroup_idx:
                descriptor[rgroup_idx[smiles]] = 1
    assert len(descriptors) == len(matched_scores), f"Number of descriptors({len(descriptors)}) doesn't match number of matcved scores({len(matched_scores)})"
    logger.info('Ridge Regressing...')
    lm = Ridge()
    lm.fit(descriptors, matched_scores)
    preds = lm.predict(descriptors)
    r2 = r2_score(matched_scores, preds)
    logger.info(f'R2 {r2}')
    logger.info(f'Intercept = {lm.intercept_:.2f}')
    for sidechains in rgroups.values():
        for rgroup in sidechains:
            rgroup.count = rgroup_counts[rgroup.smiles]
            rgroup.coefficient = lm.coef_[rgroup_idx[rgroup.smiles]]
            rgroup.idx = rgroup_idx[rgroup.smiles]
    return FreeWilsonDecomposition(rgroups, rgroup_idx, lm, r2, descriptors, decomposition, num_mols, num_reconstructed)