import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _save_other_tags(mol, fragment, mcs, orig_mol, subgraph, args):
    if args.save_counts_tag is not None:
        if not mcs:
            line = '-1 -1 -1'
        elif mcs.num_atoms == 0:
            line = '0 0 0'
        else:
            line = '1 %d %d' % (mcs.num_atoms, mcs.num_bonds)
        mol.SetProp(args.save_counts_tag, line)
    if args.save_smiles_tag is not None:
        if mcs and mcs.num_atoms > 0:
            smiles = Chem.MolToSmiles(fragment)
        else:
            smiles = '-'
        mol.SetProp(args.save_smiles_tag, smiles)
    if args.save_smarts_tag is not None:
        if mcs and mcs.num_atoms > 0:
            smarts = mcs.smarts
        else:
            smarts = '-'
        mol.SetProp(args.save_smarts_tag, smarts)