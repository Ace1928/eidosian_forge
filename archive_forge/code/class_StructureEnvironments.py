from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
class StructureEnvironments(MSONable):
    """Class used to store the chemical environments of a given structure."""
    AC = AdditionalConditions()

    class NeighborsSet:
        """Class used to store a given set of neighbors of a given site (based on the detailed_voronoi)."""

        def __init__(self, structure: Structure, isite, detailed_voronoi, site_voronoi_indices, sources=None):
            """Constructor for NeighborsSet.

            Args:
                structure: Structure object.
                isite: Index of the site for which neighbors are stored in this NeighborsSet.
                detailed_voronoi: Corresponding DetailedVoronoiContainer object containing all the possible
                    neighbors of the give site.
                site_voronoi_indices: Indices of the voronoi sites in the DetailedVoronoiContainer object that
                    make up this NeighborsSet.
                sources: Sources for this NeighborsSet, i.e. how this NeighborsSet was generated.
            """
            self.structure = structure
            self.isite = isite
            self.detailed_voronoi = detailed_voronoi
            self.voronoi = detailed_voronoi.voronoi_list2[isite]
            if (n_dupes := (len(set(site_voronoi_indices)) - len(site_voronoi_indices))):
                raise ValueError(f'Set of neighbors contains {n_dupes} duplicates!')
            self.site_voronoi_indices = sorted(site_voronoi_indices)
            if sources is None:
                self.sources = [{'origin': 'UNKNOWN'}]
            elif isinstance(sources, list):
                self.sources = sources
            else:
                self.sources = [sources]

        def get_neighb_voronoi_indices(self, permutation):
            """Get indices in the detailed_voronoi corresponding to the current permutation.

            Args:
                permutation: Current permutation for which the indices in the detailed_voronoi are needed.

            Returns:
                list[int]: indices in the detailed_voronoi.
            """
            return [self.site_voronoi_indices[ii] for ii in permutation]

        @property
        def neighb_coords(self):
            """Coordinates of neighbors for this NeighborsSet."""
            return [self.voronoi[inb]['site'].coords for inb in self.site_voronoi_indices]

        @property
        def neighb_coordsOpt(self):
            """Optimized access to the coordinates of neighbors for this NeighborsSet."""
            return self.detailed_voronoi.voronoi_list_coords[self.isite].take(self.site_voronoi_indices, axis=0)

        @property
        def neighb_sites(self):
            """Neighbors for this NeighborsSet as pymatgen Sites."""
            return [self.voronoi[inb]['site'] for inb in self.site_voronoi_indices]

        @property
        def neighb_sites_and_indices(self):
            """List of neighbors for this NeighborsSet as pymatgen Sites and their index in the original structure."""
            return [{'site': self.voronoi[inb]['site'], 'index': self.voronoi[inb]['index']} for inb in self.site_voronoi_indices]

        @property
        def coords(self):
            """Coordinates of the current central atom and its neighbors for this NeighborsSet."""
            coords = [self.structure[self.isite].coords]
            coords.extend(self.neighb_coords)
            return coords

        @property
        def normalized_distances(self):
            """Normalized distances to each neighbor in this NeighborsSet."""
            return [self.voronoi[inb]['normalized_distance'] for inb in self.site_voronoi_indices]

        @property
        def normalized_angles(self):
            """Normalized angles for each neighbor in this NeighborsSet."""
            return [self.voronoi[inb]['normalized_angle'] for inb in self.site_voronoi_indices]

        @property
        def distances(self):
            """Distances to each neighbor in this NeighborsSet."""
            return [self.voronoi[inb]['distance'] for inb in self.site_voronoi_indices]

        @property
        def angles(self):
            """Angles for each neighbor in this NeighborsSet."""
            return [self.voronoi[inb]['angle'] for inb in self.site_voronoi_indices]

        @property
        def info(self):
            """Summarized information about this NeighborsSet."""
            was = self.normalized_angles
            wds = self.normalized_distances
            angles = self.angles
            distances = self.distances
            return {'normalized_angles': was, 'normalized_distances': wds, 'normalized_angles_sum': np.sum(was), 'normalized_angles_mean': np.mean(was), 'normalized_angles_std': np.std(was), 'normalized_angles_min': np.min(was), 'normalized_angles_max': np.max(was), 'normalized_distances_mean': np.mean(wds), 'normalized_distances_std': np.std(wds), 'normalized_distances_min': np.min(wds), 'normalized_distances_max': np.max(wds), 'angles': angles, 'distances': distances, 'angles_sum': np.sum(angles), 'angles_mean': np.mean(angles), 'angles_std': np.std(angles), 'angles_min': np.min(angles), 'angles_max': np.max(angles), 'distances_mean': np.mean(distances), 'distances_std': np.std(distances), 'distances_min': np.min(distances), 'distances_max': np.max(distances)}

        def distance_plateau(self):
            """Returns the distances plateau's for this NeighborsSet."""
            all_nbs_normalized_distances_sorted = sorted((nb['normalized_distance'] for nb in self.voronoi), reverse=True)
            maxdist = np.max(self.normalized_distances)
            plateau = None
            for idist, dist in enumerate(all_nbs_normalized_distances_sorted):
                if np.isclose(dist, maxdist, rtol=0.0, atol=self.detailed_voronoi.normalized_distance_tolerance):
                    plateau = np.inf if idist == 0 else all_nbs_normalized_distances_sorted[idist - 1] - maxdist
                    break
            if plateau is None:
                raise ValueError('Plateau not found ...')
            return plateau

        def angle_plateau(self):
            """Returns the angles plateau's for this NeighborsSet."""
            all_nbs_normalized_angles_sorted = sorted((nb['normalized_angle'] for nb in self.voronoi))
            minang = np.min(self.normalized_angles)
            for nb in self.voronoi:
                print(nb)
            plateau = None
            for iang, ang in enumerate(all_nbs_normalized_angles_sorted):
                if np.isclose(ang, minang, rtol=0.0, atol=self.detailed_voronoi.normalized_angle_tolerance):
                    plateau = minang if iang == 0 else minang - all_nbs_normalized_angles_sorted[iang - 1]
                    break
            if plateau is None:
                raise ValueError('Plateau not found ...')
            return plateau

        def voronoi_grid_surface_points(self, additional_condition=1, other_origins='DO_NOTHING'):
            """
            Get the surface points in the Voronoi grid for this neighbor from the sources.
            The general shape of the points should look like a staircase such as in the following figure :

               ^
            0.0|
               |
               |      B----C
               |      |    |
               |      |    |
            a  |      k    D-------E
            n  |      |            |
            g  |      |            |
            l  |      |            |
            e  |      j            F----n---------G
               |      |                           |
               |      |                           |
               |      A----g-------h----i---------H
               |
               |
            1.0+------------------------------------------------->
              1.0              distance              2.0   ->+Inf

            Args:
                additional_condition: Additional condition for the neighbors.
                other_origins: What to do with sources that do not come from the Voronoi grid (e.g. "from hints").
            """
            src_list = []
            for src in self.sources:
                if src['origin'] == 'dist_ang_ac_voronoi':
                    if src['ac'] != additional_condition:
                        continue
                    src_list.append(src)
                else:
                    if other_origins == 'DO_NOTHING':
                        continue
                    raise NotImplementedError('Nothing implemented for other sources ...')
            if len(src_list) == 0:
                return None
            dists = [src['dp_dict']['min'] for src in src_list]
            angles = [src['ap_dict']['max'] for src in src_list]
            next_dists = [src['dp_dict']['next'] for src in src_list]
            next_angles = [src['ap_dict']['next'] for src in src_list]
            points_dict = {}
            p_dists = []
            pangs = []
            for idx in range(len(src_list)):
                if not any(np.isclose(p_dists, dists[idx])):
                    p_dists.append(dists[idx])
                if not any(np.isclose(p_dists, next_dists[idx])):
                    p_dists.append(next_dists[idx])
                if not any(np.isclose(pangs, angles[idx])):
                    pangs.append(angles[idx])
                if not any(np.isclose(pangs, next_angles[idx])):
                    pangs.append(next_angles[idx])
                d1_indices = np.argwhere(np.isclose(p_dists, dists[idx])).flatten()
                if len(d1_indices) != 1:
                    raise ValueError('Distance parameter not found ...')
                d2_indices = np.argwhere(np.isclose(p_dists, next_dists[idx])).flatten()
                if len(d2_indices) != 1:
                    raise ValueError('Distance parameter not found ...')
                a1_indices = np.argwhere(np.isclose(pangs, angles[idx])).flatten()
                if len(a1_indices) != 1:
                    raise ValueError('Angle parameter not found ...')
                a2_indices = np.argwhere(np.isclose(pangs, next_angles[idx])).flatten()
                if len(a2_indices) != 1:
                    raise ValueError('Angle parameter not found ...')
                id1 = d1_indices[0]
                id2 = d2_indices[0]
                ia1 = a1_indices[0]
                ia2 = a2_indices[0]
                for id_ia in [(id1, ia1), (id1, ia2), (id2, ia1), (id2, ia2)]:
                    points_dict.setdefault(id_ia, 0)
                    points_dict[id_ia] += 1
            new_pts = []
            for pt, pt_nb in points_dict.items():
                if pt_nb % 2 == 1:
                    new_pts.append(pt)
            sorted_points = [(0, 0)]
            move_ap_index = True
            while True:
                last_pt = sorted_points[-1]
                if move_ap_index:
                    idp = last_pt[0]
                    iap = None
                    for pt in new_pts:
                        if pt[0] == idp and pt != last_pt:
                            iap = pt[1]
                            break
                else:
                    idp = None
                    iap = last_pt[1]
                    for pt in new_pts:
                        if pt[1] == iap and pt != last_pt:
                            idp = pt[0]
                            break
                if (idp, iap) == (0, 0):
                    break
                if (idp, iap) in sorted_points:
                    raise ValueError('Error sorting points ...')
                sorted_points.append((idp, iap))
                move_ap_index = not move_ap_index
            return [(p_dists[idp], pangs[iap]) for idp, iap in sorted_points]

        @property
        def source(self):
            """
            Returns the source of this NeighborsSet (how it was generated, e.g. from which Voronoi
            cutoffs, or from hints).
            """
            if len(self.sources) != 1:
                raise RuntimeError('Number of sources different from 1 !')
            return self.sources[0]

        def add_source(self, source):
            """
            Add a source to this NeighborsSet.

            Args:
                source: Information about the generation of this NeighborsSet.
            """
            if source not in self.sources:
                self.sources.append(source)

        def __len__(self):
            return len(self.site_voronoi_indices)

        def __hash__(self) -> int:
            return len(self.site_voronoi_indices)

        def __eq__(self, other: object) -> bool:
            needed_attrs = ('isite', 'site_voronoi_indices')
            if not all((hasattr(other, attr) for attr in needed_attrs)):
                return NotImplemented
            return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs))

        def __str__(self):
            out = f'Neighbors Set for site #{self.isite} :\n'
            out += f' - Coordination number : {len(self)}\n'
            voro_indices = ', '.join((f'{site_voronoi_index}' for site_voronoi_index in self.site_voronoi_indices))
            out += f' - Voronoi indices : {voro_indices}\n'
            return out

        def as_dict(self):
            """A JSON-serializable dict representation of the NeighborsSet."""
            return {'isite': self.isite, 'site_voronoi_indices': self.site_voronoi_indices, 'sources': self.sources}

        @classmethod
        def from_dict(cls, dct, structure: Structure, detailed_voronoi) -> Self:
            """
            Reconstructs the NeighborsSet algorithm from its JSON-serializable dict representation, together with
            the structure and the DetailedVoronoiContainer.

            As an inner (nested) class, the NeighborsSet is not supposed to be used anywhere else that inside the
            StructureEnvironments. The from_dict method is thus using the structure and  detailed_voronoi when
            reconstructing itself. These two are both in the StructureEnvironments object.

            Args:
                dct: a JSON-serializable dict representation of a NeighborsSet.
                structure: The structure.
                detailed_voronoi: The Voronoi object containing all the neighboring atoms from which the subset of
                    neighbors for this NeighborsSet is extracted.

            Returns:
                NeighborsSet
            """
            return cls(structure=structure, isite=dct['isite'], detailed_voronoi=detailed_voronoi, site_voronoi_indices=dct['site_voronoi_indices'], sources=dct['sources'])

    def __init__(self, voronoi, valences, sites_map, equivalent_sites, ce_list, structure, neighbors_sets=None, info=None):
        """
        Constructor for the StructureEnvironments object.

        Args:
            voronoi: VoronoiContainer object for the structure.
            valences: Valences provided.
            sites_map: Mapping of equivalent sites to the nonequivalent sites that have been computed.
            equivalent_sites: List of list of equivalent sites of the structure.
            ce_list: List of chemical environments.
            structure: Structure object.
            neighbors_sets: List of neighbors sets.
            info: Additional information for this StructureEnvironments object.
        """
        self.voronoi = voronoi
        self.valences = valences
        self.sites_map = sites_map
        self.equivalent_sites = equivalent_sites
        self.ce_list = ce_list
        self.structure = structure
        if neighbors_sets is None:
            self.neighbors_sets = [None] * len(self.structure)
        else:
            self.neighbors_sets = neighbors_sets
        self.info = info

    def init_neighbors_sets(self, isite, additional_conditions=None, valences=None):
        """
        Initialize the list of neighbors sets for the current site.

        Args:
            isite: Index of the site under consideration.
            additional_conditions: Additional conditions to be used for the initialization of the list of
                neighbors sets, e.g. "Only anion-cation bonds", ...
            valences: List of valences for each site in the structure (needed if an additional condition based on the
                valence is used, e.g. only anion-cation bonds).
        """
        site_voronoi = self.voronoi.voronoi_list2[isite]
        if site_voronoi is None:
            return
        if additional_conditions is None:
            additional_conditions = self.AC.ALL
        if (self.AC.ONLY_ACB in additional_conditions or self.AC.ONLY_ACB_AND_NO_E2SEB) and valences is None:
            raise ChemenvError('StructureEnvironments', 'init_neighbors_sets', 'Valences are not given while only_anion_cation_bonds are allowed. Cannot continue')
        site_distance_parameters = self.voronoi.neighbors_normalized_distances[isite]
        site_angle_parameters = self.voronoi.neighbors_normalized_angles[isite]
        distance_conditions = []
        for idp, dp_dict in enumerate(site_distance_parameters):
            distance_conditions.append([])
            for inb in range(len(site_voronoi)):
                cond = inb in dp_dict['nb_indices']
                distance_conditions[idp].append(cond)
        angle_conditions = []
        for iap, ap_dict in enumerate(site_angle_parameters):
            angle_conditions.append([])
            for inb in range(len(site_voronoi)):
                cond = inb in ap_dict['nb_indices']
                angle_conditions[iap].append(cond)
        precomputed_additional_conditions = {ac: [] for ac in additional_conditions}
        for voro_nb_dict in site_voronoi:
            for ac in additional_conditions:
                cond = self.AC.check_condition(condition=ac, structure=self.structure, parameters={'valences': valences, 'neighbor_index': voro_nb_dict['index'], 'site_index': isite})
                precomputed_additional_conditions[ac].append(cond)
        for idp, dp_dict in enumerate(site_distance_parameters):
            for iap, ap_dict in enumerate(site_angle_parameters):
                for iac, ac in enumerate(additional_conditions):
                    src = {'origin': 'dist_ang_ac_voronoi', 'idp': idp, 'iap': iap, 'dp_dict': dp_dict, 'ap_dict': ap_dict, 'iac': iac, 'ac': ac, 'ac_name': self.AC.CONDITION_DESCRIPTION[ac]}
                    site_voronoi_indices = [inb for inb, _voro_nb_dict in enumerate(site_voronoi) if distance_conditions[idp][inb] and angle_conditions[iap][inb] and precomputed_additional_conditions[ac][inb]]
                    nb_set = self.NeighborsSet(structure=self.structure, isite=isite, detailed_voronoi=self.voronoi, site_voronoi_indices=site_voronoi_indices, sources=src)
                    self.add_neighbors_set(isite=isite, nb_set=nb_set)

    def add_neighbors_set(self, isite, nb_set):
        """
        Adds a neighbor set to the list of neighbors sets for this site.

        Args:
            isite: Index of the site under consideration.
            nb_set: NeighborsSet to be added.
        """
        if self.neighbors_sets[isite] is None:
            self.neighbors_sets[isite] = {}
            self.ce_list[isite] = {}
        cn = len(nb_set)
        if cn not in self.neighbors_sets[isite]:
            self.neighbors_sets[isite][cn] = []
            self.ce_list[isite][cn] = []
        try:
            nb_set_index = self.neighbors_sets[isite][cn].index(nb_set)
            self.neighbors_sets[isite][cn][nb_set_index].add_source(nb_set.source)
        except ValueError:
            self.neighbors_sets[isite][cn].append(nb_set)
            self.ce_list[isite][cn].append(None)

    def update_coordination_environments(self, isite, cn, nb_set, ce):
        """
        Updates the coordination environment for this site, coordination and neighbor set.

        Args:
            isite: Index of the site to be updated.
            cn: Coordination to be updated.
            nb_set: Neighbors set to be updated.
            ce: ChemicalEnvironments object for this neighbors set.
        """
        if self.ce_list[isite] is None:
            self.ce_list[isite] = {}
        if cn not in self.ce_list[isite]:
            self.ce_list[isite][cn] = []
        try:
            nb_set_index = self.neighbors_sets[isite][cn].index(nb_set)
        except ValueError:
            raise ValueError('Neighbors set not found in the structure environments')
        if nb_set_index == len(self.ce_list[isite][cn]):
            self.ce_list[isite][cn].append(ce)
        elif nb_set_index < len(self.ce_list[isite][cn]):
            self.ce_list[isite][cn][nb_set_index] = ce
        else:
            raise ValueError('Neighbors set not yet in ce_list !')

    def update_site_info(self, isite, info_dict):
        """
        Update information about this site.

        Args:
            isite: Index of the site for which info has to be updated.
            info_dict: Dictionary of information to be added for this site.
        """
        if 'sites_info' not in self.info:
            self.info['sites_info'] = [{} for _ in range(len(self.structure))]
        self.info['sites_info'][isite].update(info_dict)

    def get_coordination_environments(self, isite, cn, nb_set):
        """
        Get the ChemicalEnvironments for a given site, coordination and neighbors set.

        Args:
            isite: Index of the site for which the ChemicalEnvironments is looked for.
            cn: Coordination for which the ChemicalEnvironments is looked for.
            nb_set: Neighbors set for which the ChemicalEnvironments is looked for.

        Returns:
            ChemicalEnvironments
        """
        if self.ce_list[isite] is None:
            return None
        if cn not in self.ce_list[isite]:
            return None
        try:
            nb_set_index = self.neighbors_sets[isite][cn].index(nb_set)
        except ValueError:
            return None
        return self.ce_list[isite][cn][nb_set_index]

    def get_csm(self, isite, mp_symbol):
        """
        Get the continuous symmetry measure for a given site in the given coordination environment.

        Args:
            isite: Index of the site.
            mp_symbol: Symbol of the coordination environment for which we want the continuous symmetry measure.

        Returns:
            Continuous symmetry measure of the given site in the given environment.
        """
        csms = self.get_csms(isite, mp_symbol)
        if len(csms) != 1:
            raise ChemenvError('StructureEnvironments', 'get_csm', f'Number of csms for site #{isite} with mp_symbol {mp_symbol!r} = {len(csms)}')
        return csms[0]

    def get_csms(self, isite, mp_symbol) -> list:
        """
        Returns the continuous symmetry measure(s) of site with index isite with respect to the
        perfect coordination environment with mp_symbol. For some environments, a given mp_symbol might not
        be available (if there is no voronoi parameters leading to a number of neighbors corresponding to
        the coordination number of environment mp_symbol). For some environments, a given mp_symbol might
        lead to more than one csm (when two or more different voronoi parameters lead to different neighbors
        but with same number of neighbors).

        Args:
            isite: Index of the site.
            mp_symbol: MP symbol of the perfect environment for which the csm has to be given.

        Returns:
            list[CSM]: for site isite with respect to geometry mp_symbol
        """
        cn = symbol_cn_mapping[mp_symbol]
        if cn not in self.ce_list[isite]:
            return []
        return [envs[mp_symbol] for envs in self.ce_list[isite][cn]]

    def plot_csm_and_maps(self, isite, max_csm=8.0):
        """
        Plotting of the coordination numbers of a given site for all the distfactor/angfactor parameters. If the
        chemical environments are given, a color map is added to the plot, with the lowest continuous symmetry measure
        as the value for the color of that distfactor/angfactor set.

        Args:
            isite: Index of the site for which the plot has to be done
            max_csm: Maximum continuous symmetry measure to be shown.
        """
        fig = self.get_csm_and_maps(isite=isite, max_csm=max_csm)
        if fig is None:
            return
        plt.show()
        return

    def get_csm_and_maps(self, isite, max_csm=8.0, figsize=None, symmetry_measure_type=None) -> tuple[plt.Figure, plt.Axes] | None:
        """
        Plotting of the coordination numbers of a given site for all the distfactor/angfactor parameters. If the
        chemical environments are given, a color map is added to the plot, with the lowest continuous symmetry measure
        as the value for the color of that distfactor/angfactor set.

        Args:
            isite: Index of the site for which the plot has to be done.
            max_csm: Maximum continuous symmetry measure to be shown.
            figsize: Size of the figure.
            symmetry_measure_type: Type of continuous symmetry measure to be used.

        Returns:
            Matplotlib figure and axes representing the CSM and maps.
        """
        if symmetry_measure_type is None:
            symmetry_measure_type = 'csm_wcs_ctwcc'
        fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, hspace=0.0, wspace=0.0)
        ax = fig.add_subplot(gs[:])
        ax_distang = ax.twinx()
        idx = 0
        cn_maps = []
        all_wds = []
        all_was = []
        max_wd = 0.0
        for cn, nb_sets in self.neighbors_sets[isite].items():
            for inb_set, nb_set in enumerate(nb_sets):
                ce = self.ce_list[isite][cn][inb_set]
                if ce is None:
                    continue
                min_geoms = ce.minimum_geometries(max_csm=max_csm)
                if len(min_geoms) == 0:
                    continue
                wds = nb_set.normalized_distances
                max_wd = max(max_wd, *wds)
                all_wds.append(wds)
                all_was.append(nb_set.normalized_angles)
                for mp_symbol, cg_dict in min_geoms:
                    csm = cg_dict['other_symmetry_measures'][symmetry_measure_type]
                    ax.plot(idx, csm, 'ob')
                    ax.annotate(mp_symbol, xy=(idx, csm))
                cn_maps.append((cn, inb_set))
                idx += 1
        if max_wd < 1.225:
            ymax_wd = 1.25
            yticks_wd = np.linspace(1.0, ymax_wd, 6)
        elif max_wd < 1.36:
            ymax_wd = 1.4
            yticks_wd = np.linspace(1.0, ymax_wd, 5)
        elif max_wd < 1.45:
            ymax_wd = 1.5
            yticks_wd = np.linspace(1.0, ymax_wd, 6)
        elif max_wd < 1.55:
            ymax_wd = 1.6
            yticks_wd = np.linspace(1.0, ymax_wd, 7)
        elif max_wd < 1.75:
            ymax_wd = 1.8
            yticks_wd = np.linspace(1.0, ymax_wd, 5)
        elif max_wd < 1.95:
            ymax_wd = 2.0
            yticks_wd = np.linspace(1.0, ymax_wd, 6)
        elif max_wd < 2.35:
            ymax_wd = 2.5
            yticks_wd = np.linspace(1.0, ymax_wd, 7)
        else:
            ymax_wd = np.ceil(1.1 * max_wd)
            yticks_wd = np.linspace(1.0, ymax_wd, 6)
        yticks_wa = np.linspace(0.0, 1.0, 6)
        frac_bottom = 0.05
        frac_top = 0.05
        frac_middle = 0.1
        yamin = frac_bottom
        yamax = 0.5 - frac_middle / 2
        ydmin = 0.5 + frac_middle / 2
        ydmax = 1.0 - frac_top

        def yang(wa):
            return (yamax - yamin) * np.array(wa) + yamin

        def ydist(wd):
            return (np.array(wd) - 1.0) / (ymax_wd - 1.0) * (ydmax - ydmin) + ydmin
        for idx, was in enumerate(all_was):
            ax_distang.plot(0.2 + idx * np.ones_like(was), yang(was), '<g')
            alpha = 0.3 if np.mod(idx, 2) == 0 else 0.1
            ax_distang.fill_between([-0.5 + idx, 0.5 + idx], [1.0, 1.0], 0.0, facecolor='k', alpha=alpha, zorder=-1000)
        for idx, wds in enumerate(all_wds):
            ax_distang.plot(0.2 + idx * np.ones_like(wds), ydist(wds), 'sm')
        ax_distang.plot([-0.5, len(cn_maps)], [0.5, 0.5], 'k--', alpha=0.5)
        yticks = yang(yticks_wa).tolist()
        yticks.extend(ydist(yticks_wd).tolist())
        yticklabels = yticks_wa.tolist()
        yticklabels.extend(yticks_wd.tolist())
        ax_distang.set_yticks(yticks)
        ax_distang.set_yticklabels(yticklabels)
        fake_subplot_ang = fig.add_subplot(gs[1], frame_on=False)
        fake_subplot_dist = fig.add_subplot(gs[0], frame_on=False)
        fake_subplot_ang.set_yticks([])
        fake_subplot_dist.set_yticks([])
        fake_subplot_ang.set_xticks([])
        fake_subplot_dist.set_xticks([])
        fake_subplot_ang.set_ylabel('Angle parameter', labelpad=45, rotation=-90)
        fake_subplot_dist.set_ylabel('Distance parameter', labelpad=45, rotation=-90)
        fake_subplot_ang.yaxis.set_label_position('right')
        fake_subplot_dist.yaxis.set_label_position('right')
        ax_distang.set_ylim([0.0, 1.0])
        ax.set_xticks(range(len(cn_maps)))
        ax.set_ylabel('Continuous symmetry measure')
        ax.set_xlim([-0.5, len(cn_maps) - 0.5])
        ax_distang.set_xlim([-0.5, len(cn_maps) - 0.5])
        ax.set_xticklabels([str(cn_map) for cn_map in cn_maps])
        return (fig, ax)

    def get_environments_figure(self, isite, plot_type=None, title='Coordination numbers', max_dist=2.0, colormap=None, figsize=None, strategy=None):
        """
        Plotting of the coordination environments of a given site for all the distfactor/angfactor regions. The
        chemical environments with the lowest continuous symmetry measure is shown for each distfactor/angfactor
        region as the value for the color of that distfactor/angfactor region (using a colormap).

        Args:
            isite: Index of the site for which the plot has to be done.
            plot_type: How to plot the coordinations.
            title: Title for the figure.
            max_dist: Maximum distance to be plotted when the plotting of the distance is set to 'initial_normalized'
                or 'initial_real' (Warning: this is not the same meaning in both cases! In the first case, the
                closest atom lies at a "normalized" distance of 1.0 so that 2.0 means refers to this normalized
                distance while in the second case, the real distance is used).
            colormap: Color map to be used for the continuous symmetry measure.
            figsize: Size of the figure.
            strategy: Whether to plot information about one of the Chemenv Strategies.

        Returns:
            tuple[plt.Figure, plt.Axes]: matplotlib figure and axes representing the environments.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if plot_type is None:
            plot_type = {'distance_parameter': ('initial_normalized', None), 'angle_parameter': ('initial_normalized_inverted', None)}
        clr_map = cm.jet if colormap is None else colormap
        clr_min = 0.0
        clr_max = 10.0
        norm = Normalize(vmin=clr_min, vmax=clr_max)
        scalarmap = cm.ScalarMappable(norm=norm, cmap=clr_map)
        dist_limits = [1.0, max_dist]
        ang_limits = [0.0, 1.0]
        if plot_type['distance_parameter'][0] == 'one_minus_inverse_alpha_power_n':
            if plot_type['distance_parameter'][1] is None:
                exponent = 3
            else:
                exponent = plot_type['distance_parameter'][1]['exponent']
            xlabel = f'Distance parameter : $1.0-\\frac{{1.0}}{{\\alpha^{{{exponent}}}}}$'

            def dp_func(dp):
                return 1.0 - 1.0 / np.power(dp, exponent)
        elif plot_type['distance_parameter'][0] == 'initial_normalized':
            xlabel = 'Distance parameter : $\\alpha$'

            def dp_func(dp):
                return dp
        else:
            raise ValueError(f'Wrong value for distance parameter plot type "{plot_type['distance_parameter'][0]}"')
        if plot_type['angle_parameter'][0] == 'one_minus_gamma':
            ylabel = 'Angle parameter : $1.0-\\gamma$'

            def ap_func(ap):
                return 1.0 - ap
        elif plot_type['angle_parameter'][0] in ['initial_normalized_inverted', 'initial_normalized']:
            ylabel = 'Angle parameter : $\\gamma$'

            def ap_func(ap):
                return ap
        else:
            raise ValueError(f'Wrong value for angle parameter plot type "{plot_type['angle_parameter'][0]}"')
        dist_limits = [dp_func(dp) for dp in dist_limits]
        ang_limits = [ap_func(ap) for ap in ang_limits]
        for cn, cn_nb_sets in self.neighbors_sets[isite].items():
            for inb_set, nb_set in enumerate(cn_nb_sets):
                nb_set_surface_pts = nb_set.voronoi_grid_surface_points()
                if nb_set_surface_pts is None:
                    continue
                ce = self.ce_list[isite][cn][inb_set]
                if ce is None:
                    color = 'w'
                    inv_color = 'k'
                    text = f'{cn}'
                else:
                    mingeom = ce.minimum_geometry()
                    if mingeom is not None:
                        mp_symbol = mingeom[0]
                        csm = mingeom[1]['symmetry_measure']
                        color = scalarmap.to_rgba(csm)
                        inv_color = [1.0 - color[0], 1.0 - color[1], 1.0 - color[2], 1.0]
                        text = f'{mp_symbol}'
                    else:
                        color = 'w'
                        inv_color = 'k'
                        text = f'{cn}'
                nb_set_surface_pts = [(dp_func(pt[0]), ap_func(pt[1])) for pt in nb_set_surface_pts]
                polygon = Polygon(nb_set_surface_pts, closed=True, edgecolor='k', facecolor=color, linewidth=1.2)
                ax.add_patch(polygon)
                ipt = len(nb_set_surface_pts) / 2
                if ipt != int(ipt):
                    raise RuntimeError('Uneven number of surface points')
                ipt = int(ipt)
                patch_center = ((nb_set_surface_pts[0][0] + min(nb_set_surface_pts[ipt][0], dist_limits[1])) / 2, (nb_set_surface_pts[0][1] + nb_set_surface_pts[ipt][1]) / 2)
                if np.abs(nb_set_surface_pts[-1][1] - nb_set_surface_pts[-2][1]) > 0.06 and np.abs(min(nb_set_surface_pts[-1][0], dist_limits[1]) - nb_set_surface_pts[0][0]) > 0.125:
                    xytext = ((min(nb_set_surface_pts[-1][0], dist_limits[1]) + nb_set_surface_pts[0][0]) / 2, (nb_set_surface_pts[-1][1] + nb_set_surface_pts[-2][1]) / 2)
                    ax.annotate(text, xy=xytext, ha='center', va='center', color=inv_color, fontsize='x-small')
                elif np.abs(nb_set_surface_pts[ipt][1] - nb_set_surface_pts[0][1]) > 0.1 and np.abs(min(nb_set_surface_pts[ipt][0], dist_limits[1]) - nb_set_surface_pts[0][0]) > 0.125:
                    xytext = patch_center
                    ax.annotate(text, xy=xytext, ha='center', va='center', color=inv_color, fontsize='x-small')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        dist_limits.sort()
        ang_limits.sort()
        ax.set_xlim(dist_limits)
        ax.set_ylim(ang_limits)
        if strategy is not None:
            try:
                strategy.add_strategy_visualization_to_subplot(subplot=ax)
            except Exception:
                pass
        if plot_type['angle_parameter'][0] == 'initial_normalized_inverted':
            ax.axes.invert_yaxis()
        scalarmap.set_array([clr_min, clr_max])
        cb = fig.colorbar(scalarmap, ax=ax, extend='max')
        cb.set_label('Continuous symmetry measure')
        return (fig, ax)

    def plot_environments(self, isite, plot_type=None, title='Coordination numbers', max_dist=2.0, figsize=None, strategy=None):
        """
        Plotting of the coordination numbers of a given site for all the distfactor/angfactor parameters. If the
        chemical environments are given, a color map is added to the plot, with the lowest continuous symmetry measure
        as the value for the color of that distfactor/angfactor set.

        Args:
            isite: Index of the site for which the plot has to be done.
            plot_type: How to plot the coordinations.
            title: Title for the figure.
            max_dist: Maximum distance to be plotted when the plotting of the distance is set to 'initial_normalized'
                or 'initial_real' (Warning: this is not the same meaning in both cases! In the first case, the
                closest atom lies at a "normalized" distance of 1.0 so that 2.0 means refers to this normalized
                distance while in the second case, the real distance is used).
            figsize: Size of the figure.
            strategy: Whether to plot information about one of the Chemenv Strategies.
        """
        fig, _ax = self.get_environments_figure(isite=isite, plot_type=plot_type, title=title, max_dist=max_dist, figsize=figsize, strategy=strategy)
        if fig is None:
            return
        fig.show()

    def save_environments_figure(self, isite, imagename='image.png', plot_type=None, title='Coordination numbers', max_dist=2.0, figsize=None):
        """
        Saves the environments figure to a given file.

        Args:
            isite: Index of the site for which the plot has to be done.
            imagename: Name of the file to which the figure has to be saved.
            plot_type: How to plot the coordinations.
            title: Title for the figure.
            max_dist: Maximum distance to be plotted when the plotting of the distance is set to 'initial_normalized'
                or 'initial_real' (Warning: this is not the same meaning in both cases! In the first case, the
                closest atom lies at a "normalized" distance of 1.0 so that 2.0 means refers to this normalized
                distance while in the second case, the real distance is used).
            figsize: Size of the figure.
        """
        fig, _ax = self.get_environments_figure(isite=isite, plot_type=plot_type, title=title, max_dist=max_dist, figsize=figsize)
        if fig is None:
            return
        fig.savefig(imagename)

    def differences_wrt(self, other):
        """
        Return differences found in the current StructureEnvironments with respect to another StructureEnvironments.

        Args:
            other: A StructureEnvironments object.

        Returns:
            List of differences between the two StructureEnvironments objects.
        """
        differences = []
        if self.structure != other.structure:
            differences += ({'difference': 'structure', 'comparison': '__eq__', 'self': self.structure, 'other': other.structure}, {'difference': 'PREVIOUS DIFFERENCE IS DISMISSIVE', 'comparison': 'differences_wrt'})
            return differences
        if self.valences != other.valences:
            differences.append({'difference': 'valences', 'comparison': '__eq__', 'self': self.valences, 'other': other.valences})
        if self.info != other.info:
            differences.append({'difference': 'info', 'comparison': '__eq__', 'self': self.info, 'other': other.info})
        if self.voronoi != other.voronoi:
            if self.voronoi.is_close_to(other.voronoi):
                differences += ({'difference': 'voronoi', 'comparison': '__eq__', 'self': self.voronoi, 'other': other.voronoi}, {'difference': 'PREVIOUS DIFFERENCE IS DISMISSIVE', 'comparison': 'differences_wrt'})
                return differences
            differences += ({'difference': 'voronoi', 'comparison': 'is_close_to', 'self': self.voronoi, 'other': other.voronoi}, {'difference': 'PREVIOUS DIFFERENCE IS DISMISSIVE', 'comparison': 'differences_wrt'})
            return differences
        for isite, self_site_nb_sets in enumerate(self.neighbors_sets):
            other_site_nb_sets = other.neighbors_sets[isite]
            if self_site_nb_sets is None:
                if other_site_nb_sets is None:
                    continue
                differences.append({'difference': f'neighbors_sets[isite={isite!r}]', 'comparison': 'has_neighbors', 'self': 'None', 'other': set(other_site_nb_sets)})
                continue
            if other_site_nb_sets is None:
                differences.append({'difference': f'neighbors_sets[isite={isite!r}]', 'comparison': 'has_neighbors', 'self': set(self_site_nb_sets), 'other': 'None'})
                continue
            self_site_cns = set(self_site_nb_sets)
            other_site_cns = set(other_site_nb_sets)
            if self_site_cns != other_site_cns:
                differences.append({'difference': f'neighbors_sets[isite={isite!r}]', 'comparison': 'coordination_numbers', 'self': self_site_cns, 'other': other_site_cns})
            common_cns = self_site_cns.intersection(other_site_cns)
            for cn in common_cns:
                other_site_cn_nb_sets = other_site_nb_sets[cn]
                self_site_cn_nb_sets = self_site_nb_sets[cn]
                set_self_site_cn_nb_sets = set(self_site_cn_nb_sets)
                set_other_site_cn_nb_sets = set(other_site_cn_nb_sets)
                if set_self_site_cn_nb_sets != set_other_site_cn_nb_sets:
                    differences.append({'difference': f'neighbors_sets[isite={isite!r}][cn={cn!r}]', 'comparison': 'neighbors_sets', 'self': self_site_cn_nb_sets, 'other': other_site_cn_nb_sets})
                common_nb_sets = set_self_site_cn_nb_sets.intersection(set_other_site_cn_nb_sets)
                for nb_set in common_nb_sets:
                    inb_set_self = self_site_cn_nb_sets.index(nb_set)
                    inb_set_other = other_site_cn_nb_sets.index(nb_set)
                    self_ce = self.ce_list[isite][cn][inb_set_self]
                    other_ce = other.ce_list[isite][cn][inb_set_other]
                    if self_ce != other_ce:
                        if self_ce.is_close_to(other_ce):
                            differences.append({'difference': f'ce_list[isite={isite!r}][cn={cn!r}][inb_set={inb_set_self}]', 'comparison': '__eq__', 'self': self_ce, 'other': other_ce})
                        else:
                            differences.append({'difference': f'ce_list[isite={isite!r}][cn={cn!r}][inb_set={inb_set_self}]', 'comparison': 'is_close_to', 'self': self_ce, 'other': other_ce})
        return differences

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructureEnvironments):
            return NotImplemented
        if len(self.ce_list) != len(other.ce_list):
            return False
        if self.voronoi != other.voronoi:
            return False
        if len(self.valences) != len(other.valences):
            return False
        if self.sites_map != other.sites_map:
            return False
        if self.equivalent_sites != other.equivalent_sites:
            return False
        if self.structure != other.structure:
            return False
        if self.info != other.info:
            return False
        for isite, site_ces in enumerate(self.ce_list):
            site_nb_sets_self = self.neighbors_sets[isite]
            site_nb_sets_other = other.neighbors_sets[isite]
            if site_nb_sets_self != site_nb_sets_other:
                return False
            if site_ces != other.ce_list[isite]:
                return False
        return True

    def as_dict(self):
        """
        Bson-serializable dict representation of the StructureEnvironments object.

        Returns:
            Bson-serializable dict representation of the StructureEnvironments object.
        """
        ce_list_dict = [{str(cn): [ce.as_dict() if ce is not None else None for ce in ce_dict[cn]] for cn in ce_dict} if ce_dict is not None else None for ce_dict in self.ce_list]
        nbs_sets_dict = [{str(cn): [nb_set.as_dict() for nb_set in nb_sets] for cn, nb_sets in site_nbs_sets.items()} if site_nbs_sets is not None else None for site_nbs_sets in self.neighbors_sets]
        info_dict = {key: val for key, val in self.info.items() if key != 'sites_info'}
        info_dict['sites_info'] = [{'nb_sets_info': {str(cn): {str(inb_set): nb_set_info for inb_set, nb_set_info in cn_sets.items()} for cn, cn_sets in site_info['nb_sets_info'].items()}, 'time': site_info['time']} if 'nb_sets_info' in site_info else {} for site_info in self.info['sites_info']]
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'voronoi': self.voronoi.as_dict(), 'valences': self.valences, 'sites_map': self.sites_map, 'equivalent_sites': [[ps.as_dict() for ps in psl] for psl in self.equivalent_sites], 'ce_list': ce_list_dict, 'structure': self.structure.as_dict(), 'neighbors_sets': nbs_sets_dict, 'info': info_dict}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the StructureEnvironments object from a dict representation of the StructureEnvironments created
        using the as_dict method.

        Args:
            dct: dict representation of the StructureEnvironments object.

        Returns:
            StructureEnvironments object.
        """
        ce_list = [None if ce_dict == 'None' or ce_dict is None else {int(cn): [None if ced is None or ced == 'None' else ChemicalEnvironments.from_dict(ced) for ced in ce_dict[cn]] for cn in ce_dict} for ce_dict in dct['ce_list']]
        voronoi = DetailedVoronoiContainer.from_dict(dct['voronoi'])
        structure = Structure.from_dict(dct['structure'])
        neighbors_sets = [{int(cn): [cls.NeighborsSet.from_dict(nb_set_dict, structure=structure, detailed_voronoi=voronoi) for nb_set_dict in nb_sets] for cn, nb_sets in site_nbs_sets_dict.items()} if site_nbs_sets_dict is not None else None for site_nbs_sets_dict in dct['neighbors_sets']]
        info = {key: val for key, val in dct['info'].items() if key != 'sites_info'}
        if 'sites_info' in dct['info']:
            info['sites_info'] = [{'nb_sets_info': {int(cn): {int(inb_set): nb_set_info for inb_set, nb_set_info in cn_sets.items()} for cn, cn_sets in site_info['nb_sets_info'].items()}, 'time': site_info['time']} if 'nb_sets_info' in site_info else {} for site_info in dct['info']['sites_info']]
        return cls(voronoi=voronoi, valences=dct['valences'], sites_map=dct['sites_map'], equivalent_sites=[[PeriodicSite.from_dict(psd) for psd in psl] for psl in dct['equivalent_sites']], ce_list=ce_list, structure=structure, neighbors_sets=neighbors_sets, info=info)