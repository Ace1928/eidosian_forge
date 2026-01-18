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
def _get_exchange_df(self):
    """
        Loop over all sites in a graph and count the number and types of
        nearest neighbor interactions, computing +-|S_i . S_j| to construct
        a Heisenberg Hamiltonian for each graph. Sets self.ex_mat instance variable.

        TODO Deal with large variance in |S| across configs
        """
    sgraphs = self.sgraphs
    tol = self.tol
    unique_site_ids = self.unique_site_ids
    nn_interactions = self.nn_interactions
    dists = self.dists
    columns = ['E', 'E0']
    for k0, v0 in nn_interactions.items():
        for idx, j in v0.items():
            c = f'{idx}-{j}-{k0}'
            c_rev = f'{j}-{idx}-{k0}'
            if c not in columns and c_rev not in columns:
                columns.append(c)
    n_sgraphs = len(sgraphs)
    columns = columns[:n_sgraphs + 1]
    n_nn_j = len(columns) - 1
    j_columns = [name for name in columns if name not in ['E', 'E0']]
    ex_mat_empty = pd.DataFrame(columns=columns)
    ex_mat = ex_mat_empty.copy()
    if len(j_columns) < 2:
        self.ex_mat = ex_mat
    else:
        sgraphs_copy = copy.deepcopy(sgraphs)
        sgraph_index = 0
        for _graph in sgraphs:
            sgraph = sgraphs_copy.pop(0)
            ex_row = pd.DataFrame(np.zeros((1, n_nn_j + 1)), index=[sgraph_index], columns=columns)
            for idx, _node in enumerate(sgraph.graph.nodes):
                s_i = sgraph.structure.site_properties['magmom'][idx]
                for k, v in unique_site_ids.items():
                    if idx in k:
                        i_index = v
                connections = sgraph.get_connected_sites(idx)
                for connection in connections:
                    j_site = connection[2]
                    dist = round(connection[-1], 2)
                    s_j = sgraph.structure.site_properties['magmom'][j_site]
                    for k, v in unique_site_ids.items():
                        if j_site in k:
                            j_index = v
                    if abs(dist - dists['nn']) <= tol:
                        order = '-nn'
                    elif abs(dist - dists['nnn']) <= tol:
                        order = '-nnn'
                    elif abs(dist - dists['nnnn']) <= tol:
                        order = '-nnnn'
                    j_ij = f'{i_index}-{j_index}{order}'
                    j_ji = f'{j_index}-{i_index}{order}'
                    if j_ij in ex_mat.columns:
                        ex_row.loc[sgraph_index, j_ij] -= s_i * s_j
                    elif j_ji in ex_mat.columns:
                        ex_row.loc[sgraph_index, j_ji] -= s_i * s_j
            temp_df = pd.concat([ex_mat, ex_row], ignore_index=True)
            if temp_df[j_columns].equals(temp_df[j_columns].drop_duplicates(keep='first')):
                e_index = self.ordered_structures.index(sgraph.structure)
                ex_row.loc[sgraph_index, 'E'] = self.energies[e_index]
                sgraph_index += 1
                ex_mat = pd.concat([ex_mat, ex_row], ignore_index=True)
        ex_mat[j_columns] = ex_mat[j_columns].div(2)
        ex_mat[['E0']] = 1
        zeros = list((ex_mat == 0).all(axis=0))
        if True in zeros:
            c = ex_mat.columns[zeros.index(True)]
            ex_mat = ex_mat.drop(columns=[c], axis=1)
        ex_mat = ex_mat[:ex_mat.shape[1] - 1]
        self.ex_mat = ex_mat