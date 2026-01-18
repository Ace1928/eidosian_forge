from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _get_nn_dict(self):
    """Sets self.nn_interactions and self.dists instance variables describing unique
        nearest neighbor interactions.
        """
    tol = self.tol
    sgraph = self.sgraphs[0]
    unique_site_ids = self.unique_site_ids
    nn_dict = {}
    nnn_dict = {}
    nnnn_dict = {}
    all_dists = []
    for k in unique_site_ids:
        i = k[0]
        i_key = unique_site_ids[k]
        connected_sites = sgraph.get_connected_sites(i)
        dists = [round(cs[-1], 2) for cs in connected_sites]
        dists = sorted(set(dists))
        dists = dists[:3]
        all_dists += dists
    all_dists = sorted(set(all_dists))
    rm_list = []
    for idx, d in enumerate(all_dists[:-1], start=1):
        if abs(d - all_dists[idx]) < tol:
            rm_list.append(idx)
    all_dists = [d for idx, d in enumerate(all_dists) if idx not in rm_list]
    if len(all_dists) < 3:
        all_dists += [0] * (3 - len(all_dists))
    all_dists = all_dists[:3]
    labels = ('nn', 'nnn', 'nnnn')
    dists = dict(zip(labels, all_dists))
    for k in unique_site_ids:
        i = k[0]
        i_key = unique_site_ids[k]
        connected_sites = sgraph.get_connected_sites(i)
        for cs in connected_sites:
            dist = round(cs[-1], 2)
            j = cs[2]
            for key, value in unique_site_ids.items():
                if j in key:
                    j_key = value
            if abs(dist - dists['nn']) <= tol:
                nn_dict[i_key] = j_key
            elif abs(dist - dists['nnn']) <= tol:
                nnn_dict[i_key] = j_key
            elif abs(dist - dists['nnnn']) <= tol:
                nnnn_dict[i_key] = j_key
    nn_interactions = {'nn': nn_dict, 'nnn': nnn_dict, 'nnnn': nnnn_dict}
    self.dists = dists
    self.nn_interactions = nn_interactions