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
def FWBuild(fw: FreeWilsonDecomposition, pred_filter=None, mw_filter=None, hvy_filter=None, mol_filter=None) -> Generator[FreeWilsonPrediction, None, None]:
    """Enumerate the freewilson decomposition and return their predictions

       :param fw: FreeWilsonDecomposition generated from FWDecompose
       :param pred_filter: return True if the prediction is in a desireable range
                           e.g.  lambda pic50: pic50>=8
       :param mw_filter: return True if the enumerated molecular weight is in a desireable rrange
                           e.g. lambda mw: 150 < mw < 550
       :param hvy_filter: return True if the enumerated heavy couont is in a desireable rrange
                           e.g. lambda hvy: 10 < hvy < 50
       :param mol_filter: return True if the molecule is ok to be enumerated
                           e.g. lambda mol: -3 < Descriptors.MolLogp(mol) < 5
    """
    blocker = rdBase.BlockLogs()
    cycles = set()
    rgroups_no_cycles = defaultdict(list)
    rgroup_cycles = defaultdict(list)
    for key, rgroup in fw.rgroups.items():
        if key == 'Core':
            rgroups_no_cycles[key] = rgroup
            continue
        no_cycles = rgroups_no_cycles[key]
        for g in rgroup:
            no_cycles.append(g)
            continue
            if len(g.dummies) > 1:
                cycles.add(g.dummies)
                rgroup_cycles[g.dummies].append(g)
            else:
                no_cycles.append(g)
    logging.info('Enumerating rgroups with no broken cycles...')
    for k, v in rgroups_no_cycles.items():
        logging.info(f'\t{k}\t{len(v)}')
    rgroups = [rgroup for key, rgroup in sorted(rgroups_no_cycles.items())]
    for res in _enumerate(rgroups, fw, pred_filter=pred_filter, mw_filter=mw_filter, hvy_filter=hvy_filter, mol_filter=mol_filter):
        yield res
    indices = set()
    for k in fw.rgroups:
        if k[0] == 'R':
            indices.add(int(k[1:]))
    if cycles:
        logging.info('Enumerating rgroups with broken cycles...')
    for rgroup_indices in cycles:
        missing = indices - set(rgroup_indices)
        rgroups = {'Core': fw.rgroups['Core']}
        rgroups['R%s' % '.'.join([str(x) for x in rgroup_indices])] = rgroup_cycles[rgroup_indices]
        for m in missing:
            k = 'R%s' % m
            rgroups[k] = rgroups_no_cycles[k]
        for k, v in rgroups.items():
            logging.info(f'\t{k}\t{len(v)}')
        for res in _enumerate([rgroup for key, rgroup in sorted(rgroups.items())], fw, pred_filter=pred_filter, mw_filter=mw_filter, hvy_filter=hvy_filter, mol_filter=mol_filter):
            yield res