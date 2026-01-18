from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
class LocalGeometryFinder:
    """Main class used to find the local environments in a structure."""
    DEFAULT_BVA_DISTANCE_SCALE_FACTOR = 1.0
    BVA_DISTANCE_SCALE_FACTORS = dict(experimental=1.0, GGA_relaxed=1.015, LDA_relaxed=0.995)
    DEFAULT_SPG_ANALYZER_OPTIONS = dict(symprec=0.001, angle_tolerance=5)
    STRUCTURE_REFINEMENT_NONE = 'none'
    STRUCTURE_REFINEMENT_REFINED = 'refined'
    STRUCTURE_REFINEMENT_SYMMETRIZED = 'symmetrized'
    DEFAULT_STRATEGY = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
    PRESETS = {'DEFAULT': {'maximum_distance_factor': 2.0, 'minimum_angle_factor': 0.05, 'voronoi_normalized_distance_tolerance': 0.05, 'voronoi_normalized_angle_tolerance': 0.03, 'optimization': 2}}

    def __init__(self, permutations_safe_override: bool=False, plane_ordering_override: bool=True, plane_safe_permutations: bool=False, only_symbols=None):
        """
        Args:
            permutations_safe_override: If set to True, all permutations are tested (very time-consuming for large
            coordination numbers!)
            plane_ordering_override: If set to False, the ordering of the points in the plane is disabled
            plane_safe_permutations: Whether to use safe permutations.
            only_symbols: Whether to restrict the list of environments to be identified.
        """
        self.allcg = AllCoordinationGeometries(permutations_safe_override=permutations_safe_override, only_symbols=only_symbols)
        self.permutations_safe_override = permutations_safe_override
        self.plane_ordering_override = plane_ordering_override
        self.plane_safe_permutations = plane_safe_permutations
        self.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True, bva_distance_scale_factor=None, structure_refinement=self.STRUCTURE_REFINEMENT_NONE)

    def setup_parameters(self, centering_type='standard', include_central_site_in_centroid=False, bva_distance_scale_factor=None, structure_refinement=STRUCTURE_REFINEMENT_REFINED, spg_analyzer_options=None):
        """
        Setup of the parameters for the coordination geometry finder. A reference point for the geometries has to be
        chosen. This can be the centroid of the structure (including or excluding the atom for which the coordination
        geometry is looked for) or the atom itself. In the 'standard' centering_type, the reference point is the central
        atom for coordination numbers 1, 2, 3 and 4 and the centroid for coordination numbers > 4.

        Args:
            centering_type: Type of the reference point (centering) 'standard', 'centroid' or 'central_site'
            include_central_site_in_centroid: In case centering_type is 'centroid', the central site is included if
                this value is set to True.
            bva_distance_scale_factor: Scaling factor for the bond valence analyzer (this might be different whether
                the structure is an experimental one, an LDA or a GGA relaxed one, or any other relaxation scheme (where
                under- or over-estimation of bond lengths is known).
            structure_refinement: Refinement of the structure. Can be "none", "refined" or "symmetrized".
            spg_analyzer_options: Options for the SpaceGroupAnalyzer (dictionary specifying "symprec"
                and "angle_tolerance". See pymatgen's SpaceGroupAnalyzer for more information.
        """
        self.centering_type = centering_type
        self.include_central_site_in_centroid = include_central_site_in_centroid
        if bva_distance_scale_factor is not None:
            self.bva_distance_scale_factor = bva_distance_scale_factor
        else:
            self.bva_distance_scale_factor = self.DEFAULT_BVA_DISTANCE_SCALE_FACTOR
        self.structure_refinement = structure_refinement
        if spg_analyzer_options is None:
            self.spg_analyzer_options = self.DEFAULT_SPG_ANALYZER_OPTIONS
        else:
            self.spg_analyzer_options = spg_analyzer_options

    def setup_parameter(self, parameter, value):
        """
        Setup of one specific parameter to the given value. The other parameters are unchanged. See setup_parameters
        method for the list of possible parameters

        Args:
            parameter: Parameter to setup/update
            value: Value of the parameter.
        """
        self.__dict__[parameter] = value

    def setup_structure(self, structure: Structure):
        """
        Sets up the structure for which the coordination geometries have to be identified. The structure is analyzed
        with the space group analyzer and a refined structure is used

        Args:
            structure: A pymatgen Structure.
        """
        self.initial_structure = structure.copy()
        if self.structure_refinement == self.STRUCTURE_REFINEMENT_NONE:
            self.structure = structure.copy()
            self.spg_analyzer = self.symmetrized_structure = None
        else:
            self.spg_analyzer = SpacegroupAnalyzer(self.initial_structure, symprec=self.spg_analyzer_options['symprec'], angle_tolerance=self.spg_analyzer_options['angle_tolerance'])
            if self.structure_refinement == self.STRUCTURE_REFINEMENT_REFINED:
                self.structure = self.spg_analyzer.get_refined_structure()
                self.symmetrized_structure = None
            elif self.structure_refinement == self.STRUCTURE_REFINEMENT_SYMMETRIZED:
                self.structure = self.spg_analyzer.get_refined_structure()
                self.spg_analyzer_refined = SpacegroupAnalyzer(self.structure, symprec=self.spg_analyzer_options['symprec'], angle_tolerance=self.spg_analyzer_options['angle_tolerance'])
                self.symmetrized_structure = self.spg_analyzer_refined.get_symmetrized_structure()

    def get_structure(self):
        """
        Returns the pymatgen Structure that has been setup for the identification of geometries (the initial one
        might have been refined/symmetrized using the SpaceGroupAnalyzer).

        Returns:
            The pymatgen Structure that has been setup for the identification of geometries (the initial one
        might have been refined/symmetrized using the SpaceGroupAnalyzer).
        """
        return self.structure

    def set_structure(self, lattice: Lattice, species, coords, coords_are_cartesian):
        """
        Sets up the pymatgen structure for which the coordination geometries have to be identified starting from the
        lattice, the species and the coordinates

        Args:
            lattice: The lattice of the structure
            species: The species on the sites
            coords: The coordinates of the sites
            coords_are_cartesian: If set to True, the coordinates are given in Cartesian coordinates.
        """
        self.setup_structure(Structure(lattice, species, coords, coords_are_cartesian))

    def compute_coordination_environments(self, structure, indices=None, only_cations=True, strategy=DEFAULT_STRATEGY, valences='bond-valence-analysis', initial_structure_environments=None):
        """
        Args:
            structure:
            indices:
            only_cations:
            strategy:
            valences:
            initial_structure_environments:
        """
        self.setup_structure(structure=structure)
        if valences == 'bond-valence-analysis':
            bva = BVAnalyzer()
            try:
                vals = bva.get_valences(structure=structure)
            except ValueError:
                vals = 'undefined'
        elif valences == 'undefined':
            vals = valences
        else:
            len_vals, len_sites = (len(valences), len(structure))
            if len_vals != len_sites:
                raise ValueError(f'Valences ({len_vals}) do not match the number of sites in the structure ({len_sites})')
            vals = valences
        se = self.compute_structure_environments(only_cations=only_cations, only_indices=indices, valences=vals, initial_structure_environments=initial_structure_environments)
        lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
        return lse.coordination_environments

    def compute_structure_environments(self, excluded_atoms=None, only_atoms=None, only_cations=True, only_indices=None, maximum_distance_factor=PRESETS['DEFAULT']['maximum_distance_factor'], minimum_angle_factor=PRESETS['DEFAULT']['minimum_angle_factor'], max_cn=None, min_cn=None, only_symbols=None, valences='undefined', additional_conditions=None, info=None, timelimit=None, initial_structure_environments=None, get_from_hints=False, voronoi_normalized_distance_tolerance=PRESETS['DEFAULT']['voronoi_normalized_distance_tolerance'], voronoi_normalized_angle_tolerance=PRESETS['DEFAULT']['voronoi_normalized_angle_tolerance'], voronoi_distance_cutoff=None, recompute=None, optimization=PRESETS['DEFAULT']['optimization']):
        """
        Computes and returns the StructureEnvironments object containing all the information about the coordination
        environments in the structure

        Args:
            excluded_atoms: Atoms for which the coordination geometries does not have to be identified
            only_atoms: If not set to None, atoms for which the coordination geometries have to be identified
            only_cations: If set to True, will only compute environments for cations
            only_indices: If not set to None, will only compute environments the atoms of the given indices
            maximum_distance_factor: If not set to None, neighbors beyond
                maximum_distance_factor*closest_neighbor_distance are not considered
            minimum_angle_factor: If not set to None, neighbors for which the angle is lower than
                minimum_angle_factor*largest_angle_neighbor are not considered
            max_cn: maximum coordination number to be considered
            min_cn: minimum coordination number to be considered
            only_symbols: if not set to None, consider only coordination environments with the given symbols
            valences: valences of the atoms
            additional_conditions: additional conditions to be considered in the bonds (example : only bonds
                between cation and anion
            info: additional info about the calculation
            timelimit: time limit (in secs) after which the calculation of the StructureEnvironments object stops
            initial_structure_environments: initial StructureEnvironments object (most probably incomplete)
            get_from_hints: whether to add neighbors sets from "hints" (e.g. capped environment => test the
                neighbors without the cap)
            voronoi_normalized_distance_tolerance: tolerance for the normalized distance used to distinguish
                neighbors sets
            voronoi_normalized_angle_tolerance: tolerance for the normalized angle used to distinguish
                neighbors sets
            voronoi_distance_cutoff: determines distance of considered neighbors. Especially important to increase it
                for molecules in a box.
            recompute: whether to recompute the sites already computed (when initial_structure_environments
                is not None)
            optimization: optimization algorithm

        Returns:
            The StructureEnvironments object containing all the information about the coordination
            environments in the structure.
        """
        time_init = time.process_time()
        if info is None:
            info = {}
        info.update(local_geometry_finder={'parameters': {'centering_type': self.centering_type, 'include_central_site_in_centroid': self.include_central_site_in_centroid, 'structure_refinement': self.structure_refinement, 'spg_analyzer_options': self.spg_analyzer_options}})
        if only_symbols is not None:
            self.allcg = AllCoordinationGeometries(permutations_safe_override=self.permutations_safe_override, only_symbols=only_symbols)
        if valences == 'undefined':
            first_site = self.structure[0]
            try:
                sp = first_site.specie
                if isinstance(sp, Species):
                    self.valences = [int(site.specie.oxi_state) for site in self.structure]
                else:
                    self.valences = valences
            except AttributeError:
                self.valences = valences
        else:
            self.valences = valences
        self.equivalent_sites = [[site] for site in self.structure]
        self.struct_sites_to_irreducible_site_list_map = list(range(len(self.structure)))
        self.sites_map = list(range(len(self.structure)))
        indices = list(range(len(self.structure)))
        if only_cations and self.valences != 'undefined':
            sites_indices = [idx for idx in indices if self.valences[idx] >= 0]
        else:
            sites_indices = list(indices)
        if only_atoms is not None:
            sites_indices = [idx for idx in sites_indices if any((at in [sp.symbol for sp in self.structure[idx].species] for at in only_atoms))]
        if excluded_atoms:
            sites_indices = [idx for idx in sites_indices if not any((at in [sp.symbol for sp in self.structure[idx].species] for at in excluded_atoms))]
        if only_indices is not None:
            sites_indices = [isite for isite in indices if isite in only_indices]
        logging.debug('Getting DetailedVoronoiContainer')
        if voronoi_normalized_distance_tolerance is None:
            normalized_distance_tolerance = DetailedVoronoiContainer.default_normalized_distance_tolerance
        else:
            normalized_distance_tolerance = voronoi_normalized_distance_tolerance
        if voronoi_normalized_angle_tolerance is None:
            normalized_angle_tolerance = DetailedVoronoiContainer.default_normalized_angle_tolerance
        else:
            normalized_angle_tolerance = voronoi_normalized_angle_tolerance
        if voronoi_distance_cutoff is None:
            voronoi_distance_cutoff = DetailedVoronoiContainer.default_voronoi_cutoff
        self.detailed_voronoi = DetailedVoronoiContainer(self.structure, isites=sites_indices, valences=self.valences, maximum_distance_factor=maximum_distance_factor, minimum_angle_factor=minimum_angle_factor, additional_conditions=additional_conditions, normalized_distance_tolerance=normalized_distance_tolerance, normalized_angle_tolerance=normalized_angle_tolerance, voronoi_cutoff=voronoi_distance_cutoff)
        logging.debug('DetailedVoronoiContainer has been set up')
        if initial_structure_environments is not None:
            struct_envs = initial_structure_environments
            if struct_envs.structure != self.structure:
                raise ValueError('Structure is not the same in initial_structure_environments')
            if struct_envs.voronoi != self.detailed_voronoi:
                if self.detailed_voronoi.is_close_to(struct_envs.voronoi):
                    self.detailed_voronoi = struct_envs.voronoi
                else:
                    raise ValueError('Detailed Voronoi is not the same in initial_structure_environments')
            struct_envs.info = info
        else:
            struct_envs = StructureEnvironments(voronoi=self.detailed_voronoi, valences=self.valences, sites_map=self.sites_map, equivalent_sites=self.equivalent_sites, ce_list=[None] * len(self.structure), structure=self.structure, info=info)
        if min_cn is None:
            min_cn = 1
        if max_cn is None:
            max_cn = 20
        all_cns = range(min_cn, max_cn + 1)
        do_recompute = False
        if recompute is not None:
            if 'cns' in recompute:
                cns_to_recompute = recompute['cns']
                all_cns = list(set(all_cns).intersection(cns_to_recompute))
            do_recompute = True
        max_time_one_site = 0.0
        break_it = False
        if optimization > 0:
            self.detailed_voronoi.local_planes = [None] * len(self.structure)
            self.detailed_voronoi.separations = [None] * len(self.structure)
        for isite, site in enumerate(self.structure):
            if isite not in sites_indices:
                logging.debug(f' ... in site #{isite}/{len(self.structure)} ({site.species_string}) : skipped')
                continue
            if break_it:
                logging.debug(f' ... in site #{isite}/{len(self.structure)} ({site.species_string}) : skipped (timelimit)')
                continue
            logging.debug(f' ... in site #{isite}/{len(self.structure)} ({site.species_string})')
            t1 = time.process_time()
            if optimization > 0:
                self.detailed_voronoi.local_planes[isite] = {}
                self.detailed_voronoi.separations[isite] = {}
            struct_envs.init_neighbors_sets(isite=isite, additional_conditions=additional_conditions, valences=valences)
            to_add_from_hints = []
            nb_sets_info = {}
            for cn, nb_sets in struct_envs.neighbors_sets[isite].items():
                if cn not in all_cns:
                    continue
                for inb_set, nb_set in enumerate(nb_sets):
                    logging.debug(f'    ... getting environments for nb_set ({cn}, {inb_set})')
                    t_nbset1 = time.process_time()
                    ce = self.update_nb_set_environments(se=struct_envs, isite=isite, cn=cn, inb_set=inb_set, nb_set=nb_set, recompute=do_recompute, optimization=optimization)
                    t_nbset2 = time.process_time()
                    nb_sets_info.setdefault(cn, {})
                    nb_sets_info[cn][inb_set] = {'time': t_nbset2 - t_nbset1}
                    if get_from_hints:
                        for cg_symbol, cg_dict in ce:
                            cg = self.allcg[cg_symbol]
                            if cg.neighbors_sets_hints is None:
                                continue
                            logging.debug(f'       ... getting hints from cg with mp_symbol {cg_symbol!r} ...')
                            hints_info = {'csm': cg_dict['symmetry_measure'], 'nb_set': nb_set, 'permutation': cg_dict['permutation']}
                            for nb_sets_hints in cg.neighbors_sets_hints:
                                suggested_nb_set_voronoi_indices = nb_sets_hints.hints(hints_info)
                                for idx_new, new_nb_set_voronoi_indices in enumerate(suggested_nb_set_voronoi_indices):
                                    logging.debug(f'           hint # {idx_new}')
                                    new_nb_set = struct_envs.NeighborsSet(structure=struct_envs.structure, isite=isite, detailed_voronoi=struct_envs.voronoi, site_voronoi_indices=new_nb_set_voronoi_indices, sources={'origin': 'nb_set_hints', 'hints_type': nb_sets_hints.hints_type, 'suggestion_index': idx_new, 'cn_map_source': [cn, inb_set], 'cg_source_symbol': cg_symbol})
                                    cn_new_nb_set = len(new_nb_set)
                                    if max_cn is not None and cn_new_nb_set > max_cn:
                                        continue
                                    if min_cn is not None and cn_new_nb_set < min_cn:
                                        continue
                                    if new_nb_set in [ta['new_nb_set'] for ta in to_add_from_hints]:
                                        has_nb_set = True
                                    elif cn_new_nb_set not in struct_envs.neighbors_sets[isite]:
                                        has_nb_set = False
                                    else:
                                        has_nb_set = new_nb_set in struct_envs.neighbors_sets[isite][cn_new_nb_set]
                                    if not has_nb_set:
                                        to_add_from_hints.append({'isite': isite, 'new_nb_set': new_nb_set, 'cn_new_nb_set': cn_new_nb_set})
                                        logging.debug('              => to be computed')
                                    else:
                                        logging.debug('              => already present')
            logging.debug('    ... getting environments for nb_sets added from hints')
            for missing_nb_set_to_add in to_add_from_hints:
                struct_envs.add_neighbors_set(isite=isite, nb_set=missing_nb_set_to_add['new_nb_set'])
            for missing_nb_set_to_add in to_add_from_hints:
                isite_new_nb_set = missing_nb_set_to_add['isite']
                cn_new_nb_set = missing_nb_set_to_add['cn_new_nb_set']
                new_nb_set = missing_nb_set_to_add['new_nb_set']
                inew_nb_set = struct_envs.neighbors_sets[isite_new_nb_set][cn_new_nb_set].index(new_nb_set)
                logging.debug(f'    ... getting environments for nb_set ({cn_new_nb_set}, {inew_nb_set}) - from hints')
                t_nbset1 = time.process_time()
                self.update_nb_set_environments(se=struct_envs, isite=isite_new_nb_set, cn=cn_new_nb_set, inb_set=inew_nb_set, nb_set=new_nb_set, optimization=optimization)
                t_nbset2 = time.process_time()
                if cn not in nb_sets_info:
                    nb_sets_info[cn] = {}
                nb_sets_info[cn][inew_nb_set] = {'time': t_nbset2 - t_nbset1}
            t2 = time.process_time()
            struct_envs.update_site_info(isite=isite, info_dict={'time': t2 - t1, 'nb_sets_info': nb_sets_info})
            if timelimit is not None:
                time_elapsed = t2 - time_init
                time_left = timelimit - time_elapsed
                if time_left < 2.0 * max_time_one_site:
                    break_it = True
            max_time_one_site = max(max_time_one_site, t2 - t1)
            logging.debug(f'    ... computed in {t2 - t1:.2f} seconds')
        time_end = time.process_time()
        logging.debug(f'    ... compute_structure_environments ended in {time_end - time_init:.2f} seconds')
        return struct_envs

    def update_nb_set_environments(self, se, isite, cn, inb_set, nb_set, recompute=False, optimization=None):
        """
        Args:
            se:
            isite:
            cn:
            inb_set:
            nb_set:
            recompute:
            optimization:
        """
        ce = se.get_coordination_environments(isite=isite, cn=cn, nb_set=nb_set)
        if ce is not None and (not recompute):
            return ce
        ce = ChemicalEnvironments()
        neighb_coords = nb_set.neighb_coordsOpt if optimization == 2 else nb_set.neighb_coords
        self.setup_local_geometry(isite, coords=neighb_coords, optimization=optimization)
        if optimization > 0:
            logging.debug('Getting StructureEnvironments with optimized algorithm')
            nb_set.local_planes = {}
            nb_set.separations = {}
            cncgsm = self.get_coordination_symmetry_measures_optim(nb_set=nb_set, optimization=optimization)
        else:
            logging.debug('Getting StructureEnvironments with standard algorithm')
            cncgsm = self.get_coordination_symmetry_measures()
        for coord_geom_symb, dct in cncgsm.items():
            other_csms = {'csm_wocs_ctwocc': dct['csm_wocs_ctwocc'], 'csm_wocs_ctwcc': dct['csm_wocs_ctwcc'], 'csm_wocs_csc': dct['csm_wocs_csc'], 'csm_wcs_ctwocc': dct['csm_wcs_ctwocc'], 'csm_wcs_ctwcc': dct['csm_wcs_ctwcc'], 'csm_wcs_csc': dct['csm_wcs_csc'], 'rotation_matrix_wocs_ctwocc': dct['rotation_matrix_wocs_ctwocc'], 'rotation_matrix_wocs_ctwcc': dct['rotation_matrix_wocs_ctwcc'], 'rotation_matrix_wocs_csc': dct['rotation_matrix_wocs_csc'], 'rotation_matrix_wcs_ctwocc': dct['rotation_matrix_wcs_ctwocc'], 'rotation_matrix_wcs_ctwcc': dct['rotation_matrix_wcs_ctwcc'], 'rotation_matrix_wcs_csc': dct['rotation_matrix_wcs_csc'], 'scaling_factor_wocs_ctwocc': dct['scaling_factor_wocs_ctwocc'], 'scaling_factor_wocs_ctwcc': dct['scaling_factor_wocs_ctwcc'], 'scaling_factor_wocs_csc': dct['scaling_factor_wocs_csc'], 'scaling_factor_wcs_ctwocc': dct['scaling_factor_wcs_ctwocc'], 'scaling_factor_wcs_ctwcc': dct['scaling_factor_wcs_ctwcc'], 'scaling_factor_wcs_csc': dct['scaling_factor_wcs_csc'], 'translation_vector_wocs_ctwocc': dct['translation_vector_wocs_ctwocc'], 'translation_vector_wocs_ctwcc': dct['translation_vector_wocs_ctwcc'], 'translation_vector_wocs_csc': dct['translation_vector_wocs_csc'], 'translation_vector_wcs_ctwocc': dct['translation_vector_wcs_ctwocc'], 'translation_vector_wcs_ctwcc': dct['translation_vector_wcs_ctwcc'], 'translation_vector_wcs_csc': dct['translation_vector_wcs_csc']}
            ce.add_coord_geom(coord_geom_symb, dct['csm'], algo=dct['algo'], permutation=dct['indices'], local2perfect_map=dct['local2perfect_map'], perfect2local_map=dct['perfect2local_map'], detailed_voronoi_index={'cn': cn, 'index': inb_set}, other_symmetry_measures=other_csms, rotation_matrix=dct['rotation_matrix'], scaling_factor=dct['scaling_factor'])
        se.update_coordination_environments(isite=isite, cn=cn, nb_set=nb_set, ce=ce)
        return ce

    def setup_local_geometry(self, isite, coords, optimization=None):
        """
        Sets up the AbstractGeometry for the local geometry of site with index isite.

        Args:
            isite: Index of the site for which the local geometry has to be set up
            coords: The coordinates of the (local) neighbors.
        """
        self.local_geometry = AbstractGeometry(central_site=self.structure.cart_coords[isite], bare_coords=coords, centering_type=self.centering_type, include_central_site_in_centroid=self.include_central_site_in_centroid, optimization=optimization)

    def setup_test_perfect_environment(self, symbol, randomness=False, max_random_dist=0.1, symbol_type='mp_symbol', indices='RANDOM', random_translation='NONE', random_rotation='NONE', random_scale='NONE', points=None):
        """
        Args:
            symbol:
            randomness:
            max_random_dist:
            symbol_type:
            indices:
            random_translation:
            random_rotation:
            random_scale:
            points:
        """
        if symbol_type == 'IUPAC':
            cg = self.allcg.get_geometry_from_IUPAC_symbol(symbol)
        elif symbol_type in ('MP', 'mp_symbol'):
            cg = self.allcg.get_geometry_from_mp_symbol(symbol)
        elif symbol_type == 'CoordinationGeometry':
            cg = symbol
        else:
            raise ValueError('Wrong mp_symbol to setup coordination geometry')
        neighb_coords = []
        _points = points if points is not None else cg.points
        if randomness:
            rv = np.random.random_sample(3)
            while norm(rv) > 1.0:
                rv = np.random.random_sample(3)
            coords = [np.zeros(3, float) + max_random_dist * rv]
            for pp in _points:
                rv = np.random.random_sample(3)
                while norm(rv) > 1.0:
                    rv = np.random.random_sample(3)
                neighb_coords.append(np.array(pp) + max_random_dist * rv)
        else:
            coords = [np.zeros(3, float)]
            for pp in _points:
                neighb_coords.append(np.array(pp))
        if indices == 'RANDOM':
            shuffle(neighb_coords)
        elif indices == 'ORDERED':
            pass
        else:
            neighb_coords = [neighb_coords[ii] for ii in indices]
        if random_scale == 'RANDOM':
            scale = 0.1 * np.random.random_sample() + 0.95
        elif random_scale == 'NONE':
            scale = 1.0
        else:
            scale = random_scale
        coords = [scale * cc for cc in coords]
        neighb_coords = [scale * cc for cc in neighb_coords]
        if random_rotation == 'RANDOM':
            uu = np.random.random_sample(3) + 0.1
            uu = uu / norm(uu)
            theta = np.pi * np.random.random_sample()
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            ux = uu[0]
            uy = uu[1]
            uz = uu[2]
            rand_rot = [[ux * ux + (1.0 - ux * ux) * cos_theta, ux * uy * (1.0 - cos_theta) - uz * sin_theta, ux * uz * (1.0 - cos_theta) + uy * sin_theta], [ux * uy * (1.0 - cos_theta) + uz * sin_theta, uy * uy + (1.0 - uy * uy) * cos_theta, uy * uz * (1.0 - cos_theta) - ux * sin_theta], [ux * uz * (1.0 - cos_theta) - uy * sin_theta, uy * uz * (1.0 - cos_theta) + ux * sin_theta, uz * uz + (1.0 - uz * uz) * cos_theta]]
        elif random_rotation == 'NONE':
            rand_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            rand_rot = random_rotation
        new_coords = []
        for coord in coords:
            new_cc = np.dot(rand_rot, coord).T
            new_coords.append(new_cc.ravel())
        coords = new_coords
        new_coords = []
        for coord in neighb_coords:
            new_cc = np.dot(rand_rot, coord.T)
            new_coords.append(new_cc.ravel())
        neighb_coords = new_coords
        if random_translation == 'RANDOM':
            translation = 10.0 * (2.0 * np.random.random_sample(3) - 1.0)
        elif random_translation == 'NONE':
            translation = np.zeros(3, float)
        else:
            translation = random_translation
        coords = [cc + translation for cc in coords]
        neighb_coords = [cc + translation for cc in neighb_coords]
        coords.extend(neighb_coords)
        species = ['O'] * len(coords)
        species[0] = 'Cu'
        amin = np.min([cc[0] for cc in coords])
        amax = np.max([cc[0] for cc in coords])
        bmin = np.min([cc[1] for cc in coords])
        bmax = np.max([cc[1] for cc in coords])
        cmin = np.min([cc[2] for cc in coords])
        cmax = np.max([cc[2] for cc in coords])
        factor = 5.0
        aa = factor * max([amax - amin, bmax - bmin, cmax - cmin])
        lattice = Lattice.cubic(a=aa)
        structure = Structure(lattice=lattice, species=species, coords=coords, to_unit_cell=False, coords_are_cartesian=True)
        self.setup_structure(structure=structure)
        self.setup_local_geometry(isite=0, coords=neighb_coords)
        self.perfect_geometry = AbstractGeometry.from_cg(cg=cg)

    def setup_random_structure(self, coordination):
        """
        Sets up a purely random structure with a given coordination.

        Args:
            coordination: coordination number for the random structure.
        """
        aa = 0.4
        bb = -0.2
        coords = []
        for _ in range(coordination + 1):
            coords.append(aa * np.random.random_sample(3) + bb)
        self.set_structure(lattice=np.array(np.eye(3) * 10, float), species=['Si'] * (coordination + 1), coords=coords, coords_are_cartesian=False)
        self.setup_random_indices_local_geometry(coordination)

    def setup_random_indices_local_geometry(self, coordination):
        """
        Sets up random indices for the local geometry, for testing purposes

        Args:
            coordination: coordination of the local geometry.
        """
        self.icentral_site = 0
        self.indices = list(range(1, coordination + 1))
        np.random.shuffle(self.indices)

    def setup_ordered_indices_local_geometry(self, coordination):
        """
        Sets up ordered indices for the local geometry, for testing purposes

        Args:
            coordination: coordination of the local geometry.
        """
        self.icentral_site = 0
        self.indices = list(range(1, coordination + 1))

    def setup_explicit_indices_local_geometry(self, explicit_indices):
        """
        Sets up explicit indices for the local geometry, for testing purposes

        Args:
            explicit_indices: explicit indices for the neighbors (set of numbers
        from 0 to CN-1 in a given order).
        """
        self.icentral_site = 0
        self.indices = [ii + 1 for ii in explicit_indices]

    def get_coordination_symmetry_measures(self, only_minimum=True, all_csms=True, optimization=None):
        """
        Returns the continuous symmetry measures of the current local geometry in a dictionary.

        Returns:
            the continuous symmetry measures of the current local geometry in a dictionary.
        """
        test_geometries = self.allcg.get_implemented_geometries(len(self.local_geometry.coords))
        if len(self.local_geometry.coords) == 1:
            if len(test_geometries) == 0:
                return {}
            result_dict = {'S:1': {'csm': 0.0, 'indices': [0], 'algo': 'EXPLICIT', 'local2perfect_map': {0: 0}, 'perfect2local_map': {0: 0}, 'scaling_factor': None, 'rotation_matrix': None, 'translation_vector': None}}
            if all_csms:
                for csmtype in ['wocs_ctwocc', 'wocs_ctwcc', 'wocs_csc', 'wcs_ctwocc', 'wcs_ctwcc', 'wcs_csc']:
                    result_dict['S:1'][f'csm_{csmtype}'] = 0.0
                    result_dict['S:1'][f'scaling_factor_{csmtype}'] = None
                    result_dict['S:1'][f'rotation_matrix_{csmtype}'] = None
                    result_dict['S:1'][f'translation_vector_{csmtype}'] = None
            return result_dict
        result_dict = {}
        for geometry in test_geometries:
            self.perfect_geometry = AbstractGeometry.from_cg(cg=geometry, centering_type=self.centering_type, include_central_site_in_centroid=self.include_central_site_in_centroid)
            points_perfect = self.perfect_geometry.points_wcs_ctwcc()
            cgsm = self.coordination_geometry_symmetry_measures(geometry, points_perfect=points_perfect, optimization=optimization)
            result, permutations, algos, local2perfect_maps, perfect2local_maps = cgsm
            if only_minimum:
                if len(result) > 0:
                    imin = np.argmin([rr['symmetry_measure'] for rr in result])
                    algo = algos[imin] if geometry.algorithms is not None else algos
                    result_dict[geometry.mp_symbol] = {'csm': result[imin]['symmetry_measure'], 'indices': permutations[imin], 'algo': algo, 'local2perfect_map': local2perfect_maps[imin], 'perfect2local_map': perfect2local_maps[imin], 'scaling_factor': 1.0 / result[imin]['scaling_factor'], 'rotation_matrix': np.linalg.inv(result[imin]['rotation_matrix']), 'translation_vector': result[imin]['translation_vector']}
                    if all_csms:
                        self._update_results_all_csms(result_dict, permutations, imin, geometry)
            else:
                result_dict[geometry.mp_symbol] = {'csm': result, 'indices': permutations, 'algo': algos, 'local2perfect_map': local2perfect_maps, 'perfect2local_map': perfect2local_maps}
        return result_dict

    def _update_results_all_csms(self, result_dict, permutations, imin, geometry):
        permutation = permutations[imin]
        pdist = self.local_geometry.points_wocs_ctwocc(permutation=permutation)
        pperf = self.perfect_geometry.points_wocs_ctwocc()
        sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
        result_dict[geometry.mp_symbol]['csm_wocs_ctwocc'] = sm_info['symmetry_measure']
        result_dict[geometry.mp_symbol]['rotation_matrix_wocs_ctwocc'] = np.linalg.inv(sm_info['rotation_matrix'])
        result_dict[geometry.mp_symbol]['scaling_factor_wocs_ctwocc'] = 1.0 / sm_info['scaling_factor']
        result_dict[geometry.mp_symbol]['translation_vector_wocs_ctwocc'] = self.local_geometry.centroid_without_centre
        pdist = self.local_geometry.points_wocs_ctwcc(permutation=permutation)
        pperf = self.perfect_geometry.points_wocs_ctwcc()
        sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
        result_dict[geometry.mp_symbol]['csm_wocs_ctwcc'] = sm_info['symmetry_measure']
        result_dict[geometry.mp_symbol]['rotation_matrix_wocs_ctwcc'] = np.linalg.inv(sm_info['rotation_matrix'])
        result_dict[geometry.mp_symbol]['scaling_factor_wocs_ctwcc'] = 1.0 / sm_info['scaling_factor']
        result_dict[geometry.mp_symbol]['translation_vector_wocs_ctwcc'] = self.local_geometry.centroid_with_centre
        pdist = self.local_geometry.points_wocs_csc(permutation=permutation)
        pperf = self.perfect_geometry.points_wocs_csc()
        sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
        result_dict[geometry.mp_symbol]['csm_wocs_csc'] = sm_info['symmetry_measure']
        result_dict[geometry.mp_symbol]['rotation_matrix_wocs_csc'] = np.linalg.inv(sm_info['rotation_matrix'])
        result_dict[geometry.mp_symbol]['scaling_factor_wocs_csc'] = 1.0 / sm_info['scaling_factor']
        result_dict[geometry.mp_symbol]['translation_vector_wocs_csc'] = self.local_geometry.bare_centre
        pdist = self.local_geometry.points_wcs_ctwocc(permutation=permutation)
        pperf = self.perfect_geometry.points_wcs_ctwocc()
        sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
        result_dict[geometry.mp_symbol]['csm_wcs_ctwocc'] = sm_info['symmetry_measure']
        result_dict[geometry.mp_symbol]['rotation_matrix_wcs_ctwocc'] = np.linalg.inv(sm_info['rotation_matrix'])
        result_dict[geometry.mp_symbol]['scaling_factor_wcs_ctwocc'] = 1.0 / sm_info['scaling_factor']
        result_dict[geometry.mp_symbol]['translation_vector_wcs_ctwocc'] = self.local_geometry.centroid_without_centre
        pdist = self.local_geometry.points_wcs_ctwcc(permutation=permutation)
        pperf = self.perfect_geometry.points_wcs_ctwcc()
        sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
        result_dict[geometry.mp_symbol]['csm_wcs_ctwcc'] = sm_info['symmetry_measure']
        result_dict[geometry.mp_symbol]['rotation_matrix_wcs_ctwcc'] = np.linalg.inv(sm_info['rotation_matrix'])
        result_dict[geometry.mp_symbol]['scaling_factor_wcs_ctwcc'] = 1.0 / sm_info['scaling_factor']
        result_dict[geometry.mp_symbol]['translation_vector_wcs_ctwcc'] = self.local_geometry.centroid_with_centre
        pdist = self.local_geometry.points_wcs_csc(permutation=permutation)
        pperf = self.perfect_geometry.points_wcs_csc()
        sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
        result_dict[geometry.mp_symbol]['csm_wcs_csc'] = sm_info['symmetry_measure']
        result_dict[geometry.mp_symbol]['rotation_matrix_wcs_csc'] = np.linalg.inv(sm_info['rotation_matrix'])
        result_dict[geometry.mp_symbol]['scaling_factor_wcs_csc'] = 1.0 / sm_info['scaling_factor']
        result_dict[geometry.mp_symbol]['translation_vector_wcs_csc'] = self.local_geometry.bare_centre

    def get_coordination_symmetry_measures_optim(self, only_minimum=True, all_csms=True, nb_set=None, optimization=None):
        """
        Returns the continuous symmetry measures of the current local geometry in a dictionary.

        Returns:
            the continuous symmetry measures of the current local geometry in a dictionary.
        """
        cn = len(self.local_geometry.coords)
        test_geometries = self.allcg.get_implemented_geometries(cn)
        if all((cg.algorithms[0].algorithm_type == EXPLICIT_PERMUTATIONS for cg in test_geometries)):
            return self.get_coordination_symmetry_measures(only_minimum=only_minimum, all_csms=all_csms, optimization=optimization)
        if not all((all((algo.algorithm_type == SEPARATION_PLANE for algo in cg.algorithms)) for cg in test_geometries)):
            raise ValueError('All algorithms should be EXPLICIT_PERMUTATIONS or SEPARATION_PLANE')
        result_dict = {}
        for geometry in test_geometries:
            logging.log(level=5, msg=f'Getting Continuous Symmetry Measure with Separation Plane algorithm for geometry "{geometry.ce_symbol}"')
            self.perfect_geometry = AbstractGeometry.from_cg(cg=geometry, centering_type=self.centering_type, include_central_site_in_centroid=self.include_central_site_in_centroid)
            points_perfect = self.perfect_geometry.points_wcs_ctwcc()
            cgsm = self.coordination_geometry_symmetry_measures_sepplane_optim(geometry, points_perfect=points_perfect, nb_set=nb_set, optimization=optimization)
            result, permutations, algos, local2perfect_maps, perfect2local_maps = cgsm
            if only_minimum and len(result) > 0:
                imin = np.argmin([rr['symmetry_measure'] for rr in result])
                algo = algos[imin] if geometry.algorithms is not None else algos
                result_dict[geometry.mp_symbol] = {'csm': result[imin]['symmetry_measure'], 'indices': permutations[imin], 'algo': algo, 'local2perfect_map': local2perfect_maps[imin], 'perfect2local_map': perfect2local_maps[imin], 'scaling_factor': 1.0 / result[imin]['scaling_factor'], 'rotation_matrix': np.linalg.inv(result[imin]['rotation_matrix']), 'translation_vector': result[imin]['translation_vector']}
                if all_csms:
                    self._update_results_all_csms(result_dict, permutations, imin, geometry)
        return result_dict

    def coordination_geometry_symmetry_measures(self, coordination_geometry, tested_permutations=False, points_perfect=None, optimization=None):
        """Returns the symmetry measures of a given coordination_geometry for a set of
        permutations depending on the permutation setup. Depending on the parameters of
        the LocalGeometryFinder and on the coordination geometry, different methods are called.

        Args:
            coordination_geometry: Coordination geometry for which the symmetry measures are looked for

        Raises:
            NotImplementedError: if the permutation_setup does not exist

        Returns:
            the symmetry measures of a given coordination_geometry for a set of permutations
        """
        if tested_permutations:
            tested_permutations = set()
        if self.permutations_safe_override:
            raise ValueError('No permutations safe override anymore')
        csms = []
        permutations = []
        algos = []
        local2perfect_maps = []
        perfect2local_maps = []
        for algo in coordination_geometry.algorithms:
            if algo.algorithm_type == EXPLICIT_PERMUTATIONS:
                return self.coordination_geometry_symmetry_measures_standard(coordination_geometry, algo, points_perfect=points_perfect, optimization=optimization)
            if algo.algorithm_type == SEPARATION_PLANE:
                cgsm = self.coordination_geometry_symmetry_measures_separation_plane(coordination_geometry, algo, tested_permutations=tested_permutations, points_perfect=points_perfect)
                csm, perm, algo, local2perfect_map, perfect2local_map = cgsm
                csms.extend(csm)
                permutations.extend(perm)
                algos.extend(algo)
                local2perfect_maps.extend(local2perfect_map)
                perfect2local_maps.extend(perfect2local_map)
        return (csms, permutations, algos, local2perfect_maps, perfect2local_maps)

    def coordination_geometry_symmetry_measures_sepplane_optim(self, coordination_geometry, points_perfect=None, nb_set=None, optimization=None):
        """Returns the symmetry measures of a given coordination_geometry for a set of
        permutations depending on the permutation setup. Depending on the parameters of
        the LocalGeometryFinder and on the coordination geometry, different methods are called.

        Args:
            coordination_geometry: Coordination geometry for which the symmetry measures are looked for

        Raises:
            NotImplementedError: if the permutation_setup does not exist

        Returns:
            the symmetry measures of a given coordination_geometry for a set of permutations
        """
        csms = []
        permutations = []
        algos = []
        local2perfect_maps = []
        perfect2local_maps = []
        for algo in coordination_geometry.algorithms:
            if algo.algorithm_type == SEPARATION_PLANE:
                cgsm = self.coordination_geometry_symmetry_measures_separation_plane_optim(coordination_geometry, algo, points_perfect=points_perfect, nb_set=nb_set, optimization=optimization)
                csm, perm, algo, local2perfect_map, perfect2local_map = cgsm
                csms.extend(csm)
                permutations.extend(perm)
                algos.extend(algo)
                local2perfect_maps.extend(local2perfect_map)
                perfect2local_maps.extend(perfect2local_map)
        return (csms, permutations, algos, local2perfect_maps, perfect2local_maps)

    def coordination_geometry_symmetry_measures_standard(self, coordination_geometry, algo, points_perfect=None, optimization=None):
        """
        Returns the symmetry measures for a set of permutations (whose setup depends on the coordination geometry)
        for the coordination geometry "coordination_geometry". Standard implementation looking for the symmetry
        measures of each permutation

        Args:
            coordination_geometry: The coordination geometry to be investigated

        Returns:
            The symmetry measures for the given coordination geometry for each permutation investigated.
        """
        if optimization == 2:
            permutations_symmetry_measures = [None] * len(algo.permutations)
            permutations = []
            algos = []
            local2perfect_maps = []
            perfect2local_maps = []
            for idx, perm in enumerate(algo.permutations):
                local2perfect_map = {}
                perfect2local_map = {}
                permutations.append(perm)
                for iperfect, ii in enumerate(perm):
                    perfect2local_map[iperfect] = ii
                    local2perfect_map[ii] = iperfect
                local2perfect_maps.append(local2perfect_map)
                perfect2local_maps.append(perfect2local_map)
                points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=perm)
                sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
                sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
                permutations_symmetry_measures[idx] = sm_info
                algos.append(str(algo))
            return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)
        permutations_symmetry_measures = [None] * len(algo.permutations)
        permutations = []
        algos = []
        local2perfect_maps = []
        perfect2local_maps = []
        for idx, perm in enumerate(algo.permutations):
            local2perfect_map = {}
            perfect2local_map = {}
            permutations.append(perm)
            for iperfect, ii in enumerate(perm):
                perfect2local_map[iperfect] = ii
                local2perfect_map[ii] = iperfect
            local2perfect_maps.append(local2perfect_map)
            perfect2local_maps.append(perfect2local_map)
            points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=perm)
            sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
            sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
            permutations_symmetry_measures[idx] = sm_info
            algos.append(str(algo))
        return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)

    def coordination_geometry_symmetry_measures_separation_plane(self, coordination_geometry, separation_plane_algo, testing=False, tested_permutations=False, points_perfect=None):
        """
        Returns the symmetry measures of the given coordination geometry "coordination_geometry" using separation
        facets to reduce the complexity of the system. Caller to the refined 2POINTS, 3POINTS and other ...

        Args:
            coordination_geometry: The coordination geometry to be investigated

        Returns:
            The symmetry measures for the given coordination geometry for each plane and permutation investigated.
        """
        permutations = []
        permutations_symmetry_measures = []
        plane_separations = []
        algos = []
        perfect2local_maps = []
        local2perfect_maps = []
        if testing:
            separation_permutations = []
        nplanes = 0
        for npoints in range(separation_plane_algo.minimum_number_of_points, min(separation_plane_algo.maximum_number_of_points, 4) + 1):
            for points_combination in itertools.combinations(self.local_geometry.coords, npoints):
                if npoints == 2:
                    if collinear(points_combination[0], points_combination[1], self.local_geometry.central_site, tolerance=0.25):
                        continue
                    plane = Plane.from_3points(points_combination[0], points_combination[1], self.local_geometry.central_site)
                elif npoints == 3:
                    if collinear(points_combination[0], points_combination[1], points_combination[2], tolerance=0.25):
                        continue
                    plane = Plane.from_3points(points_combination[0], points_combination[1], points_combination[2])
                elif npoints > 3:
                    plane = Plane.from_npoints(points_combination, best_fit='least_square_distance')
                else:
                    raise ValueError('Wrong number of points to initialize separation plane')
                cgsm = self._cg_csm_separation_plane(coordination_geometry=coordination_geometry, sep_plane=separation_plane_algo, local_plane=plane, plane_separations=plane_separations, dist_tolerances=DIST_TOLERANCES, testing=testing, tested_permutations=tested_permutations, points_perfect=points_perfect)
                csm, perm, algo = (cgsm[0], cgsm[1], cgsm[2])
                if csm is not None:
                    permutations_symmetry_measures.extend(csm)
                    permutations.extend(perm)
                    for thisperm in perm:
                        p2l = {}
                        l2p = {}
                        for i_p, pp in enumerate(thisperm):
                            p2l[i_p] = pp
                            l2p[pp] = i_p
                        perfect2local_maps.append(p2l)
                        local2perfect_maps.append(l2p)
                    algos.extend(algo)
                    if testing:
                        separation_permutations.extend(cgsm[3])
                    nplanes += 1
            if nplanes > 0:
                break
        if nplanes == 0:
            return self.coordination_geometry_symmetry_measures_fallback_random(coordination_geometry, points_perfect=points_perfect)
        if testing:
            return (permutations_symmetry_measures, permutations, separation_permutations)
        return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)

    def coordination_geometry_symmetry_measures_separation_plane_optim(self, coordination_geometry, separation_plane_algo, points_perfect=None, nb_set=None, optimization=None):
        """
        Returns the symmetry measures of the given coordination geometry "coordination_geometry" using separation
        facets to reduce the complexity of the system. Caller to the refined 2POINTS, 3POINTS and other ...

        Args:
            coordination_geometry: The coordination geometry to be investigated.
            separation_plane_algo: Separation Plane algorithm used.
            points_perfect: Points corresponding to the perfect geometry.
            nb_set: Neighbor set for this set of points. (used to store already computed separation planes)
            optimization: Optimization level (1 or 2).

        Returns:
            tuple: Continuous symmetry measures for the given coordination geometry for each plane and permutation
                investigated, corresponding permutations, corresponding algorithms,
                corresponding mappings from local to perfect environment and corresponding mappings
                from perfect to local environment.
        """
        if optimization == 2:
            logging.log(level=5, msg='... using optimization = 2')
            cgcsmoptim = self._cg_csm_separation_plane_optim2
        elif optimization == 1:
            logging.log(level=5, msg='... using optimization = 2')
            cgcsmoptim = self._cg_csm_separation_plane_optim1
        else:
            raise ValueError('Optimization should be 1 or 2')
        cn = len(self.local_geometry.coords)
        permutations = []
        permutations_symmetry_measures = []
        algos = []
        perfect2local_maps = []
        local2perfect_maps = []
        if separation_plane_algo.separation in nb_set.separations:
            for local_plane, npsep in nb_set.separations[separation_plane_algo.separation].values():
                cgsm = cgcsmoptim(coordination_geometry=coordination_geometry, sepplane=separation_plane_algo, local_plane=local_plane, points_perfect=points_perfect, separation_indices=npsep)
                csm, perm, algo, _ = (cgsm[0], cgsm[1], cgsm[2], cgsm[3])
                permutations_symmetry_measures.extend(csm)
                permutations.extend(perm)
                for thisperm in perm:
                    p2l = {}
                    l2p = {}
                    for i_p, pp in enumerate(thisperm):
                        p2l[i_p] = pp
                        l2p[pp] = i_p
                    perfect2local_maps.append(p2l)
                    local2perfect_maps.append(l2p)
                algos.extend(algo)
        for npoints in range(self.allcg.minpoints[cn], min(self.allcg.maxpoints[cn], 3) + 1):
            for ipoints_combination in itertools.combinations(range(self.local_geometry.cn), npoints):
                if ipoints_combination in nb_set.local_planes:
                    continue
                nb_set.local_planes[ipoints_combination] = None
                points_combination = [self.local_geometry.coords[ip] for ip in ipoints_combination]
                if npoints == 2:
                    if collinear(points_combination[0], points_combination[1], self.local_geometry.central_site, tolerance=0.25):
                        continue
                    plane = Plane.from_3points(points_combination[0], points_combination[1], self.local_geometry.central_site)
                elif npoints == 3:
                    if collinear(points_combination[0], points_combination[1], points_combination[2], tolerance=0.25):
                        continue
                    plane = Plane.from_3points(points_combination[0], points_combination[1], points_combination[2])
                elif npoints > 3:
                    plane = Plane.from_npoints(points_combination, best_fit='least_square_distance')
                else:
                    raise ValueError('Wrong number of points to initialize separation plane')
                nb_set.local_planes[ipoints_combination] = plane
                dig = plane.distances_indices_groups(points=self.local_geometry._coords, delta_factor=0.1, sign=True)
                grouped_indices = dig[2]
                new_seps = []
                for ng in range(1, len(grouped_indices) + 1):
                    inplane = list(itertools.chain(*grouped_indices[:ng]))
                    if len(inplane) > self.allcg.maxpoints_inplane[cn]:
                        break
                    inplane = [ii[0] for ii in inplane]
                    outplane = list(itertools.chain(*grouped_indices[ng:]))
                    s1 = [ii_sign[0] for ii_sign in outplane if ii_sign[1] < 0]
                    s2 = [ii_sign[0] for ii_sign in outplane if ii_sign[1] > 0]
                    separation = sort_separation_tuple([s1, inplane, s2])
                    sep = tuple((len(gg) for gg in separation))
                    if sep not in self.allcg.separations_cg[cn]:
                        continue
                    if sep not in nb_set.separations:
                        nb_set.separations[sep] = {}
                    _sep = [np.array(ss, dtype=int) for ss in separation]
                    nb_set.separations[sep][separation] = (plane, _sep)
                    if sep == separation_plane_algo.separation:
                        new_seps.append(_sep)
                for separation_indices in new_seps:
                    cgsm = cgcsmoptim(coordination_geometry=coordination_geometry, sepplane=separation_plane_algo, local_plane=plane, points_perfect=points_perfect, separation_indices=separation_indices)
                    csm, perm, algo, _ = (cgsm[0], cgsm[1], cgsm[2], cgsm[3])
                    permutations_symmetry_measures.extend(csm)
                    permutations.extend(perm)
                    for thisperm in perm:
                        p2l = {}
                        l2p = {}
                        for i_p, pp in enumerate(thisperm):
                            p2l[i_p] = pp
                            l2p[pp] = i_p
                        perfect2local_maps.append(p2l)
                        local2perfect_maps.append(l2p)
                    algos.extend(algo)
        if len(permutations_symmetry_measures) == 0:
            return self.coordination_geometry_symmetry_measures_fallback_random(coordination_geometry, points_perfect=points_perfect)
        return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)

    def _cg_csm_separation_plane(self, coordination_geometry, sep_plane, local_plane, plane_separations, dist_tolerances=None, testing=False, tested_permutations=False, points_perfect=None):
        argref_separation = sep_plane.argsorted_ref_separation_perm
        plane_found = False
        permutations = []
        permutations_symmetry_measures = []
        if testing:
            separation_permutations = []
        dist_tolerances = dist_tolerances or DIST_TOLERANCES
        for dist_tolerance in dist_tolerances:
            algo = 'NOT_FOUND'
            separation = local_plane.indices_separate(self.local_geometry._coords, dist_tolerance)
            separation = sort_separation(separation)
            if separation_in_list(separation, plane_separations):
                continue
            if len(separation[1]) != len(sep_plane.plane_points):
                continue
            if len(separation[0]) == len(sep_plane.point_groups[0]):
                this_separation = separation
                plane_separations.append(this_separation)
            elif len(separation[0]) == len(sep_plane.point_groups[1]):
                this_separation = [list(separation[2]), list(separation[1]), list(separation[0])]
                plane_separations.append(this_separation)
            else:
                continue
            if sep_plane.ordered_plane:
                inp = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in this_separation[1]]
                if sep_plane.ordered_point_groups[0]:
                    pp_s0 = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in this_separation[0]]
                    ordind_s0 = local_plane.project_and_to2dim_ordered_indices(pp_s0)
                    sep0 = [this_separation[0][ii] for ii in ordind_s0]
                else:
                    sep0 = list(this_separation[0])
                if sep_plane.ordered_point_groups[1]:
                    pp_s2 = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in this_separation[2]]
                    ordind_s2 = local_plane.project_and_to2dim_ordered_indices(pp_s2)
                    sep2 = [this_separation[2][ii] for ii in ordind_s2]
                else:
                    sep2 = list(this_separation[2])
                separation_perm = list(sep0)
                ordind = local_plane.project_and_to2dim_ordered_indices(inp)
                separation_perm.extend([this_separation[1][ii] for ii in ordind])
                algo = 'SEPARATION_PLANE_2POINTS_ORDERED'
                separation_perm.extend(sep2)
            else:
                separation_perm = list(this_separation[0])
                separation_perm.extend(this_separation[1])
                algo = 'SEPARATION_PLANE_2POINTS'
                separation_perm.extend(this_separation[2])
            if self.plane_safe_permutations:
                sep_perms = sep_plane.safe_separation_permutations(ordered_plane=sep_plane.ordered_plane, ordered_point_groups=sep_plane.ordered_point_groups)
            else:
                sep_perms = sep_plane.permutations
            for sep_perm in sep_perms:
                perm1 = [separation_perm[ii] for ii in sep_perm]
                pp = [perm1[ii] for ii in argref_separation]
                if isinstance(tested_permutations, set) and coordination_geometry.equivalent_indices is not None:
                    tuple_ref_perm = coordination_geometry.ref_permutation(pp)
                    if tuple_ref_perm in tested_permutations:
                        continue
                    tested_permutations.add(tuple_ref_perm)
                permutations.append(pp)
                if testing:
                    separation_permutations.append(sep_perm)
                points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=pp)
                sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
                sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
                permutations_symmetry_measures.append(sm_info)
            if plane_found:
                break
        if len(permutations_symmetry_measures) > 0:
            if testing:
                return (permutations_symmetry_measures, permutations, algo, separation_permutations)
            return (permutations_symmetry_measures, permutations, [sep_plane.algorithm_type] * len(permutations))
        if plane_found:
            if testing:
                return (permutations_symmetry_measures, permutations, [], [])
            return (permutations_symmetry_measures, permutations, [])
        if testing:
            return (None, None, None, None)
        return (None, None, None)

    def _cg_csm_separation_plane_optim1(self, coordination_geometry, sepplane, local_plane, points_perfect=None, separation_indices=None):
        argref_separation = sepplane.argsorted_ref_separation_perm
        permutations = []
        permutations_symmetry_measures = []
        stop_search = False
        if sepplane.ordered_plane:
            inp = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in separation_indices[1]]
            if sepplane.ordered_point_groups[0]:
                pp_s0 = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in separation_indices[0]]
                ordind_s0 = local_plane.project_and_to2dim_ordered_indices(pp_s0)
                sep0 = [separation_indices[0][ii] for ii in ordind_s0]
            else:
                sep0 = list(separation_indices[0])
            if sepplane.ordered_point_groups[1]:
                pp_s2 = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in separation_indices[2]]
                ordind_s2 = local_plane.project_and_to2dim_ordered_indices(pp_s2)
                sep2 = [separation_indices[2][ii] for ii in ordind_s2]
            else:
                sep2 = list(separation_indices[2])
            separation_perm = list(sep0)
            ordind = local_plane.project_and_to2dim_ordered_indices(inp)
            separation_perm.extend([separation_indices[1][ii] for ii in ordind])
            separation_perm.extend(sep2)
        else:
            separation_perm = list(separation_indices[0])
            separation_perm.extend(separation_indices[1])
            separation_perm.extend(separation_indices[2])
        if self.plane_safe_permutations:
            sep_perms = sepplane.safe_separation_permutations(ordered_plane=sepplane.ordered_plane, ordered_point_groups=sepplane.ordered_point_groups)
        else:
            sep_perms = sepplane.permutations
        for sep_perm in sep_perms:
            perm1 = [separation_perm[ii] for ii in sep_perm]
            pp = [perm1[ii] for ii in argref_separation]
            permutations.append(pp)
            points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=pp)
            sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
            sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
            permutations_symmetry_measures.append(sm_info)
        if len(permutations_symmetry_measures) > 0:
            return (permutations_symmetry_measures, permutations, [sepplane.algorithm_type] * len(permutations), stop_search)
        return ([], [], [], stop_search)

    def _cg_csm_separation_plane_optim2(self, coordination_geometry, sepplane, local_plane, points_perfect=None, separation_indices=None):
        argref_separation = sepplane.argsorted_ref_separation_perm
        permutations = []
        permutations_symmetry_measures = []
        stop_search = False
        if sepplane.ordered_plane:
            inp = self.local_geometry.coords.take(separation_indices[1], axis=0)
            if sepplane.ordered_point_groups[0]:
                pp_s0 = self.local_geometry.coords.take(separation_indices[0], axis=0)
                ordind_s0 = local_plane.project_and_to2dim_ordered_indices(pp_s0)
                sep0 = separation_indices[0].take(ordind_s0)
            else:
                sep0 = separation_indices[0]
            if sepplane.ordered_point_groups[1]:
                pp_s2 = self.local_geometry.coords.take(separation_indices[2], axis=0)
                ordind_s2 = local_plane.project_and_to2dim_ordered_indices(pp_s2)
                sep2 = separation_indices[2].take(ordind_s2)
            else:
                sep2 = separation_indices[2]
            ordind = local_plane.project_and_to2dim_ordered_indices(inp)
            inp1 = separation_indices[1].take(ordind)
            separation_perm = np.concatenate((sep0, inp1, sep2))
        else:
            separation_perm = np.concatenate(separation_indices)
        if self.plane_safe_permutations:
            sep_perms = sepplane.safe_separation_permutations(ordered_plane=sepplane.ordered_plane, ordered_point_groups=sepplane.ordered_point_groups)
        else:
            sep_perms = sepplane.permutations
        for sep_perm in sep_perms:
            perm1 = separation_perm.take(sep_perm)
            pp = perm1.take(argref_separation)
            permutations.append(pp)
            points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=pp)
            sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
            sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
            permutations_symmetry_measures.append(sm_info)
        if len(permutations_symmetry_measures) > 0:
            return (permutations_symmetry_measures, permutations, [sepplane.algorithm_type] * len(permutations), stop_search)
        return ([], [], [], stop_search)

    def coordination_geometry_symmetry_measures_fallback_random(self, coordination_geometry, NRANDOM=10, points_perfect=None):
        """
        Returns the symmetry measures for a random set of permutations for the coordination geometry
        "coordination_geometry". Fallback implementation for the plane separation algorithms measures
        of each permutation

        Args:
            coordination_geometry: The coordination geometry to be investigated
            NRANDOM: Number of random permutations to be tested

        Returns:
            The symmetry measures for the given coordination geometry for each permutation investigated.
        """
        permutations_symmetry_measures = [None] * NRANDOM
        permutations = []
        algos = []
        perfect2local_maps = []
        local2perfect_maps = []
        for idx in range(NRANDOM):
            perm = np.random.permutation(coordination_geometry.coordination_number)
            permutations.append(perm)
            p2l = {}
            l2p = {}
            for i_p, pp in enumerate(perm):
                p2l[i_p] = pp
                l2p[pp] = i_p
            perfect2local_maps.append(p2l)
            local2perfect_maps.append(l2p)
            points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=perm)
            sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
            sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
            permutations_symmetry_measures[idx] = sm_info
            algos.append('APPROXIMATE_FALLBACK')
        return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)