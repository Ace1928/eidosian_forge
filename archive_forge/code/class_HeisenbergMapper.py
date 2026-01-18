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
class HeisenbergMapper:
    """Class to compute exchange parameters from low energy magnetic orderings.

    Attributes:
        strategy (object): Class from pymatgen.analysis.local_env for constructing graphs.
        sgraphs (list): StructureGraph objects.
        unique_site_ids (dict): Maps each site to its unique numerical identifier.
        wyckoff_ids (dict): Maps unique numerical identifier to wyckoff position.
        nn_interactions (dict): {i: j} pairs of NN interactions between unique sites.
        dists (dict): NN, NNN, and NNNN interaction distances
        ex_mat (DataFrame): Invertible Heisenberg Hamiltonian for each graph.
        ex_params (dict): Exchange parameter values (meV/atom)
    """

    def __init__(self, ordered_structures, energies, cutoff=0, tol: float=0.02):
        """
        Exchange parameters are computed by mapping to a classical Heisenberg
        model. Strategy is the scheme for generating neighbors. Currently only
        MinimumDistanceNN is implemented.
        n+1 unique orderings are required to compute n exchange
        parameters.

        First run a MagneticOrderingsWF to obtain low energy collinear magnetic
        orderings and find the magnetic ground state. Then enumerate magnetic
        states with the ground state as the input structure, find the subset
        of supercells that map to the ground state, and do static calculations
        for these orderings.

        Args:
            ordered_structures (list): Structure objects with magmoms.
            energies (list): Total energies of each relaxed magnetic structure.
            cutoff (float): Cutoff in Angstrom for nearest neighbor search.
                Defaults to 0 (only NN, no NNN, etc.)
            tol (float): Tolerance (in Angstrom) on nearest neighbor distances
                being equal.
        """
        self.ordered_structures_ = ordered_structures
        self.energies_ = energies
        hs = HeisenbergScreener(ordered_structures, energies, screen=False)
        ordered_structures = hs.screened_structures
        energies = hs.screened_energies
        self.ordered_structures = ordered_structures
        self.energies = energies
        self.cutoff = cutoff
        self.tol = tol
        self.sgraphs = self._get_graphs(cutoff, ordered_structures)
        self.unique_site_ids, self.wyckoff_ids = self._get_unique_sites(ordered_structures[0])
        self.nn_interactions = self.dists = self.ex_mat = self.ex_params = None
        if len(self.sgraphs) < 2:
            raise SystemExit('We need at least 2 unique orderings.')
        self._get_nn_dict()
        self._get_exchange_df()

    @staticmethod
    def _get_graphs(cutoff, ordered_structures):
        """
        Generate graph representations of magnetic structures with nearest
        neighbor bonds. Right now this only works for MinimumDistanceNN.

        Args:
            cutoff (float): Cutoff in Angstrom for nearest neighbor search.
            ordered_structures (list): Structure objects.

        Returns:
            sgraphs (list): StructureGraph objects.
        """
        strategy = MinimumDistanceNN(cutoff=cutoff, get_all_sites=True) if cutoff else MinimumDistanceNN()
        return [StructureGraph.from_local_env_strategy(s, strategy=strategy) for s in ordered_structures]

    @staticmethod
    def _get_unique_sites(structure):
        """
        Get dict that maps site indices to unique identifiers.

        Args:
            structure (Structure): ground state Structure object.

        Returns:
            unique_site_ids (dict): maps tuples of equivalent site indices to a
                unique int identifier
            wyckoff_ids (dict): maps tuples of equivalent site indices to their
                wyckoff symbols
        """
        s0 = CollinearMagneticStructureAnalyzer(structure, make_primitive=False, threshold=0.0).get_nonmagnetic_structure(make_primitive=False)
        if 'wyckoff' in s0.site_properties:
            s0.remove_site_property('wyckoff')
        symm_s0 = SpacegroupAnalyzer(s0).get_symmetrized_structure()
        wyckoff = ['n/a'] * len(symm_s0)
        equivalent_indices = symm_s0.equivalent_indices
        wyckoff_symbols = symm_s0.wyckoff_symbols
        unique_site_ids = {}
        wyckoff_ids = {}
        for idx, (indices, symbol) in enumerate(zip(equivalent_indices, wyckoff_symbols)):
            unique_site_ids[tuple(indices)] = idx
            wyckoff_ids[idx] = symbol
            for index in indices:
                wyckoff[index] = symbol
        return (unique_site_ids, wyckoff_ids)

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

    def get_exchange(self):
        """
        Take Heisenberg Hamiltonian and corresponding energy for each row and
        solve for the exchange parameters.

        Returns:
            ex_params (dict): Exchange parameter values (meV/atom).
        """
        ex_mat = self.ex_mat
        E = ex_mat[['E']]
        j_names = [j for j in ex_mat.columns if j != 'E']
        if len(j_names) < 3:
            j_avg = self.estimate_exchange()
            ex_params = {'<J>': j_avg}
            self.ex_params = ex_params
            return ex_params
        H = np.array(ex_mat.loc[:, ex_mat.columns != 'E'].values).astype(float)
        H_inv = np.linalg.inv(H)
        j_ij = np.dot(H_inv, E)
        j_ij[1:] *= 1000
        j_ij = j_ij.tolist()
        ex_params = {j_name: j[0] for j_name, j in zip(j_names, j_ij)}
        self.ex_params = ex_params
        return ex_params

    def get_low_energy_orderings(self):
        """
        Find lowest energy FM and AFM orderings to compute E_AFM - E_FM.

        Returns:
            fm_struct (Structure): fm structure with 'magmom' site property
            afm_struct (Structure): afm structure with 'magmom' site property
            fm_e (float): fm energy
            afm_e (float): afm energy
        """
        fm_struct, afm_struct = (None, None)
        mag_min = np.inf
        mag_max = 0.001
        fm_e_min = 0
        afm_e_min = 0
        for s, e in zip(self.ordered_structures, self.energies):
            ordering = CollinearMagneticStructureAnalyzer(s, threshold=0, make_primitive=False).ordering
            magmoms = s.site_properties['magmom']
            if ordering == Ordering.FM and e < fm_e_min:
                fm_struct = s
                mag_max = abs(sum(magmoms))
                fm_e = e
                fm_e_min = e
            if ordering == Ordering.AFM and e < afm_e_min:
                afm_struct = s
                afm_e = e
                mag_min = abs(sum(magmoms))
                afm_e_min = e
        if not fm_struct or not afm_struct:
            for s, e in zip(self.ordered_structures, self.energies):
                magmoms = s.site_properties['magmom']
                if abs(sum(magmoms)) > mag_max:
                    fm_struct = s
                    fm_e = e
                    mag_max = abs(sum(magmoms))
                if abs(sum(magmoms)) < mag_min:
                    afm_struct = s
                    afm_e = e
                    mag_min = abs(sum(magmoms))
                    afm_e_min = e
                elif abs(sum(magmoms)) == 0 and mag_min == 0 and (e < afm_e_min):
                    afm_struct = s
                    afm_e = e
                    afm_e_min = e
        fm_struct = CollinearMagneticStructureAnalyzer(fm_struct, make_primitive=False, threshold=0.0).get_structure_with_only_magnetic_atoms(make_primitive=False)
        afm_struct = CollinearMagneticStructureAnalyzer(afm_struct, make_primitive=False, threshold=0.0).get_structure_with_only_magnetic_atoms(make_primitive=False)
        return (fm_struct, afm_struct, fm_e, afm_e)

    def estimate_exchange(self, fm_struct=None, afm_struct=None, fm_e=None, afm_e=None):
        """
        Estimate <J> for a structure based on low energy FM and AFM orderings.

        Args:
            fm_struct (Structure): fm structure with 'magmom' site property
            afm_struct (Structure): afm structure with 'magmom' site property
            fm_e (float): fm energy/atom
            afm_e (float): afm energy/atom

        Returns:
            j_avg (float): Average exchange parameter (meV/atom)
        """
        if any((arg is None for arg in [fm_struct, afm_struct, fm_e, afm_e])):
            fm_struct, afm_struct, fm_e, afm_e = self.get_low_energy_orderings()
        magmoms = fm_struct.site_properties['magmom']
        m_avg = np.mean([np.sqrt(m ** 2) for m in magmoms])
        if m_avg < 1:
            logging.warning('Local magnetic moments are small (< 1 muB / atom). The exchange parameters may be wrong, but <J> and the mean field critical temperature estimate may be OK.')
        delta_e = afm_e - fm_e
        j_avg = delta_e / m_avg ** 2
        j_avg *= 1000
        return j_avg

    def get_mft_temperature(self, j_avg):
        """
        Crude mean field estimate of critical temperature based on <J> for
        one sublattice, or solving the coupled equations for a multi-sublattice
        material.

        Args:
            j_avg (float): j_avg (float): Average exchange parameter (meV/atom)

        Returns:
            mft_t (float): Critical temperature (K)
        """
        n_sub_lattices = len(self.unique_site_ids)
        k_boltzmann = 0.0861733
        if n_sub_lattices == 1:
            mft_t = 2 * abs(j_avg) / 3 / k_boltzmann
        else:
            omega = np.zeros((n_sub_lattices, n_sub_lattices))
            ex_params = self.ex_params
            ex_params = {k: v for k, v in ex_params.items() if k != 'E0'}
            for k in ex_params:
                sites = k.split('-')
                sites = [int(num) for num in sites[:2]]
                i, j = (sites[0], sites[1])
                omega[i, j] += ex_params[k]
                omega[j, i] += ex_params[k]
            omega = omega * 2 / 3 / k_boltzmann
            eigen_vals, _eigen_vecs = np.linalg.eig(omega)
            mft_t = max(eigen_vals)
        if mft_t > 1500:
            logging.warning('This mean field estimate is too high! Probably the true low energy orderings were not given as inputs.')
        return mft_t

    def get_interaction_graph(self, filename=None):
        """
        Get a StructureGraph with edges and weights that correspond to exchange
        interactions and J_ij values, respectively.

        Args:
            filename (str): if not None, save interaction graph to filename.

        Returns:
            igraph (StructureGraph): Exchange interaction graph.
        """
        structure = self.ordered_structures[0]
        sgraph = self.sgraphs[0]
        igraph = StructureGraph.from_empty_graph(structure, edge_weight_name='exchange_constant', edge_weight_units='meV')
        if '<J>' in self.ex_params:
            warning_msg = '\n                Only <J> is available. The interaction graph will not tell\n                you much.\n                '
            logging.warning(warning_msg)
        for i, _node in enumerate(sgraph.graph.nodes):
            connections = sgraph.get_connected_sites(i)
            for c in connections:
                jimage = c[1]
                j = c[2]
                dist = c[-1]
                j_exc = self._get_j_exc(i, j, dist)
                igraph.add_edge(i, j, to_jimage=jimage, weight=j_exc, warn_duplicates=False)
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            dumpfn(igraph, filename)
        return igraph

    def _get_j_exc(self, i, j, dist):
        """
        Convenience method for looking up exchange parameter between two sites.

        Args:
            i (int): index of ith site
            j (int): index of jth site
            dist (float): distance (Angstrom) between sites
                (10E-2 precision)

        Returns:
            j_exc (float): Exchange parameter in meV
        """
        for k, v in self.unique_site_ids.items():
            if i in k:
                i_index = v
            if j in k:
                j_index = v
        order = ''
        if abs(dist - self.dists['nn']) <= self.tol:
            order = '-nn'
        elif abs(dist - self.dists['nnn']) <= self.tol:
            order = '-nnn'
        elif abs(dist - self.dists['nnnn']) <= self.tol:
            order = '-nnnn'
        j_ij = f'{i_index}-{j_index}{order}'
        j_ji = f'{j_index}-{i_index}{order}'
        if j_ij in self.ex_params:
            j_exc = self.ex_params[j_ij]
        elif j_ji in self.ex_params:
            j_exc = self.ex_params[j_ji]
        else:
            j_exc = 0
        if '<J>' in self.ex_params and order == '-nn':
            j_exc = self.ex_params['<J>']
        return j_exc

    def get_heisenberg_model(self):
        """Save results of mapping to a HeisenbergModel object.

        Returns:
            HeisenbergModel: MSONable object.
        """
        hm_formula = str(self.ordered_structures_[0].reduced_formula)
        hm_structures = self.ordered_structures
        hm_energies = self.energies
        hm_cutoff = self.cutoff
        hm_tol = self.tol
        hm_sgraphs = self.sgraphs
        hm_usi = self.unique_site_ids
        hm_wids = self.wyckoff_ids
        hm_nni = self.nn_interactions
        hm_d = self.dists
        hm_em = self.ex_mat.to_json()
        hm_ep = self.get_exchange()
        hm_javg = self.estimate_exchange()
        hm_igraph = self.get_interaction_graph()
        return HeisenbergModel(hm_formula, hm_structures, hm_energies, hm_cutoff, hm_tol, hm_sgraphs, hm_usi, hm_wids, hm_nni, hm_d, hm_em, hm_ep, hm_javg, hm_igraph)