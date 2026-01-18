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
class LightStructureEnvironments(MSONable):
    """
    Class used to store the chemical environments of a given structure obtained from a given ChemenvStrategy. Currently,
    only strategies leading to the determination of a unique environment for each site is allowed
    This class does not store all the information contained in the StructureEnvironments object, only the coordination
    environment found.
    """
    DELTA_MAX_OXIDATION_STATE = 0.1
    DEFAULT_STATISTICS_FIELDS = ('anion_list', 'anion_atom_list', 'cation_list', 'cation_atom_list', 'neutral_list', 'neutral_atom_list', 'atom_coordination_environments_present', 'ion_coordination_environments_present', 'fraction_atom_coordination_environments_present', 'fraction_ion_coordination_environments_present', 'coordination_environments_atom_present', 'coordination_environments_ion_present')

    class NeighborsSet:
        """
        Class used to store a given set of neighbors of a given site (based on a list of sites, the voronoi
        container is not part of the LightStructureEnvironments object).
        """

        def __init__(self, structure: Structure, isite, all_nbs_sites, all_nbs_sites_indices):
            """Constructor for NeighborsSet.

            Args:
                structure: Structure object.
                isite: Index of the site for which neighbors are stored in this NeighborsSet.
                all_nbs_sites: All the possible neighbors for this site.
                all_nbs_sites_indices: Indices of the sites in all_nbs_sites that make up this NeighborsSet.
            """
            self.structure = structure
            self.isite = isite
            self.all_nbs_sites = all_nbs_sites
            indices = set(all_nbs_sites_indices)
            if len(indices) != len(all_nbs_sites_indices):
                raise ValueError('Set of neighbors contains duplicates !')
            self.all_nbs_sites_indices = sorted(indices)
            self.all_nbs_sites_indices_unsorted = all_nbs_sites_indices

        @property
        def neighb_coords(self):
            """Coordinates of neighbors for this NeighborsSet."""
            return [self.all_nbs_sites[inb]['site'].coords for inb in self.all_nbs_sites_indices_unsorted]

        @property
        def neighb_sites(self):
            """Neighbors for this NeighborsSet as pymatgen Sites."""
            return [self.all_nbs_sites[inb]['site'] for inb in self.all_nbs_sites_indices_unsorted]

        @property
        def neighb_sites_and_indices(self):
            """List of neighbors for this NeighborsSet as pymatgen Sites and their index in the original structure."""
            return [{'site': self.all_nbs_sites[inb]['site'], 'index': self.all_nbs_sites[inb]['index']} for inb in self.all_nbs_sites_indices_unsorted]

        @property
        def neighb_indices_and_images(self) -> list[dict[str, int]]:
            """List of indices and images with respect to the original unit cell sites for this NeighborsSet."""
            return [{'index': self.all_nbs_sites[inb]['index'], 'image_cell': self.all_nbs_sites[inb]['image_cell']} for inb in self.all_nbs_sites_indices_unsorted]

        def __len__(self) -> int:
            return len(self.all_nbs_sites_indices)

        def __hash__(self) -> int:
            return len(self.all_nbs_sites_indices)

        def __eq__(self, other: object) -> bool:
            needed_attrs = ('isite', 'all_nbs_sites_indices')
            if not all((hasattr(other, attr) for attr in needed_attrs)):
                return NotImplemented
            return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs))

        def __str__(self):
            return f'Neighbors Set for site #{self.isite} :\n - Coordination number : {len(self)}\n - Neighbors sites indices : {', '.join((f'{nb_idxs}' for nb_idxs in self.all_nbs_sites_indices))}\n'

        def as_dict(self):
            """A JSON-serializable dict representation of the NeighborsSet."""
            return {'isite': self.isite, 'all_nbs_sites_indices': self.all_nbs_sites_indices_unsorted}

        @classmethod
        def from_dict(cls, dct, structure: Structure, all_nbs_sites) -> Self:
            """
            Reconstructs the NeighborsSet algorithm from its JSON-serializable dict representation, together with
            the structure and all the possible neighbors sites.

            As an inner (nested) class, the NeighborsSet is not supposed to be used anywhere else that inside the
            LightStructureEnvironments. The from_dict method is thus using the structure and all_nbs_sites when
            reconstructing itself. These two are both in the LightStructureEnvironments object.

            Args:
                dct: a JSON-serializable dict representation of a NeighborsSet.
                structure: The structure.
                all_nbs_sites: The list of all the possible neighbors for a given site.

            Returns:
                NeighborsSet
            """
            return cls(structure=structure, isite=dct['isite'], all_nbs_sites=all_nbs_sites, all_nbs_sites_indices=dct['all_nbs_sites_indices'])

    def __init__(self, strategy, coordination_environments=None, all_nbs_sites=None, neighbors_sets=None, structure=None, valences=None, valences_origin=None):
        """
        Constructor for the LightStructureEnvironments object.

        Args:
            strategy: ChemEnv strategy used to get the environments.
            coordination_environments: The coordination environments identified.
            all_nbs_sites: All the possible neighbors for each site in the structure.
            neighbors_sets: The neighbors sets of each site in the structure.
            structure: The structure.
            valences: The valences used to get the environments (if needed).
            valences_origin: How the valences were obtained (e.g. from the Bond-valence analysis or from the original
                structure).
        """
        self.strategy = strategy
        self.statistics_dict = None
        self.coordination_environments = coordination_environments
        self._all_nbs_sites = all_nbs_sites
        self.neighbors_sets = neighbors_sets
        self.structure = structure
        self.valences = valences
        self.valences_origin = valences_origin

    @classmethod
    def from_structure_environments(cls, strategy, structure_environments, valences=None, valences_origin=None) -> Self:
        """
        Construct a LightStructureEnvironments object from a strategy and a StructureEnvironments object.

        Args:
            strategy: ChemEnv strategy used.
            structure_environments: StructureEnvironments object from which to construct the LightStructureEnvironments.
            valences: The valences of each site in the structure.
            valences_origin: How the valences were obtained (e.g. from the Bond-valence analysis or from the original
                structure).

        Returns:
            LightStructureEnvironments
        """
        structure = structure_environments.structure
        strategy.set_structure_environments(structure_environments=structure_environments)
        coordination_environments: list = [None] * len(structure)
        neighbors_sets: list = [None] * len(structure)
        _all_nbs_sites: list = []
        all_nbs_sites: list = []
        if valences is None:
            valences = structure_environments.valences
            if valences_origin is None:
                valences_origin = 'from_structure_environments'
        elif valences_origin is None:
            valences_origin = 'user-specified'
        for idx, site in enumerate(structure):
            site_ces_and_nbs_list = strategy.get_site_ce_fractions_and_neighbors(site, strategy_info=True)
            if site_ces_and_nbs_list is None:
                continue
            coordination_environments[idx] = []
            neighbors_sets[idx] = []
            site_ces = []
            site_nbs_sets: list = []
            for ce_and_neighbors in site_ces_and_nbs_list:
                _all_nbs_sites_indices = []
                ce_dict = {'ce_symbol': ce_and_neighbors['ce_symbol'], 'ce_fraction': ce_and_neighbors['ce_fraction']}
                if ce_and_neighbors['ce_dict'] is not None:
                    csm = ce_and_neighbors['ce_dict']['other_symmetry_measures'][strategy.symmetry_measure_type]
                else:
                    csm = None
                ce_dict['csm'] = csm
                ce_dict['permutation'] = (ce_and_neighbors.get('ce_dict') or {}).get('permutation')
                site_ces.append(ce_dict)
                neighbors = ce_and_neighbors['neighbors']
                for nb_site_and_index in neighbors:
                    nb_site = nb_site_and_index['site']
                    try:
                        n_all_nbs_sites_index = all_nbs_sites.index(nb_site)
                    except ValueError:
                        nb_index_unitcell = nb_site_and_index['index']
                        diff = nb_site.frac_coords - structure[nb_index_unitcell].frac_coords
                        rounddiff = np.round(diff)
                        if not np.allclose(diff, rounddiff):
                            raise ValueError('Weird, differences between one site in a periodic image cell is not integer ...')
                        nb_image_cell = np.array(rounddiff, int)
                        n_all_nbs_sites_index = len(_all_nbs_sites)
                        _all_nbs_sites.append({'site': nb_site, 'index': nb_index_unitcell, 'image_cell': nb_image_cell})
                        all_nbs_sites.append(nb_site)
                    _all_nbs_sites_indices.append(n_all_nbs_sites_index)
                nb_set = cls.NeighborsSet(structure=structure, isite=idx, all_nbs_sites=_all_nbs_sites, all_nbs_sites_indices=_all_nbs_sites_indices)
                site_nbs_sets.append(nb_set)
            coordination_environments[idx] = site_ces
            neighbors_sets[idx] = site_nbs_sets
        return cls(strategy=strategy, coordination_environments=coordination_environments, all_nbs_sites=_all_nbs_sites, neighbors_sets=neighbors_sets, structure=structure, valences=valences, valences_origin=valences_origin)

    def setup_statistic_lists(self):
        """Set up the statistics of environments for this LightStructureEnvironments."""
        self.statistics_dict = {'valences_origin': self.valences_origin, 'anion_list': {}, 'anion_number': None, 'anion_atom_list': {}, 'anion_atom_number': None, 'cation_list': {}, 'cation_number': None, 'cation_atom_list': {}, 'cation_atom_number': None, 'neutral_list': {}, 'neutral_number': None, 'neutral_atom_list': {}, 'neutral_atom_number': None, 'atom_coordination_environments_present': {}, 'ion_coordination_environments_present': {}, 'coordination_environments_ion_present': {}, 'coordination_environments_atom_present': {}, 'fraction_ion_coordination_environments_present': {}, 'fraction_atom_coordination_environments_present': {}, 'fraction_coordination_environments_ion_present': {}, 'fraction_coordination_environments_atom_present': {}, 'count_ion_present': {}, 'count_atom_present': {}, 'count_coordination_environments_present': {}}
        atom_stat = self.statistics_dict['atom_coordination_environments_present']
        ce_atom_stat = self.statistics_dict['coordination_environments_atom_present']
        fraction_atom_stat = self.statistics_dict['fraction_atom_coordination_environments_present']
        fraction_ce_atom_stat = self.statistics_dict['fraction_coordination_environments_atom_present']
        count_atoms = self.statistics_dict['count_atom_present']
        count_ce = self.statistics_dict['count_coordination_environments_present']
        for isite, site in enumerate(self.structure):
            site_species = []
            if self.valences != 'undefined':
                for sp, occ in site.species.items():
                    valence = self.valences[isite]
                    strspecie = str(Species(sp.symbol, valence))
                    if valence < 0:
                        specie_list = self.statistics_dict['anion_list']
                        atomlist = self.statistics_dict['anion_atom_list']
                    elif valence > 0:
                        specie_list = self.statistics_dict['cation_list']
                        atomlist = self.statistics_dict['cation_atom_list']
                    else:
                        specie_list = self.statistics_dict['neutral_list']
                        atomlist = self.statistics_dict['neutral_atom_list']
                    if strspecie not in specie_list:
                        specie_list[strspecie] = occ
                    else:
                        specie_list[strspecie] += occ
                    if sp.symbol not in atomlist:
                        atomlist[sp.symbol] = occ
                    else:
                        atomlist[sp.symbol] += occ
                    site_species.append((sp.symbol, valence, occ))
            if self.coordination_environments[isite] is not None:
                site_envs = [(ce_piece_dict['ce_symbol'], ce_piece_dict['ce_fraction']) for ce_piece_dict in self.coordination_environments[isite]]
                for ce_symbol, fraction in site_envs:
                    if fraction is None:
                        continue
                    count_ce.setdefault(ce_symbol, 0.0)
                    count_ce[ce_symbol] += fraction
                for sp, occ in site.species.items():
                    elmt = sp.symbol
                    if elmt not in atom_stat:
                        atom_stat[elmt] = {}
                        count_atoms[elmt] = 0.0
                    count_atoms[elmt] += occ
                    for ce_symbol, fraction in site_envs:
                        if fraction is None:
                            continue
                        if ce_symbol not in atom_stat[elmt]:
                            atom_stat[elmt][ce_symbol] = 0.0
                        atom_stat[elmt][ce_symbol] += occ * fraction
                        if ce_symbol not in ce_atom_stat:
                            ce_atom_stat[ce_symbol] = {}
                        if elmt not in ce_atom_stat[ce_symbol]:
                            ce_atom_stat[ce_symbol][elmt] = 0.0
                        ce_atom_stat[ce_symbol][elmt] += occ * fraction
                if self.valences != 'undefined':
                    ion_stat = self.statistics_dict['ion_coordination_environments_present']
                    ce_ion_stat = self.statistics_dict['coordination_environments_ion_present']
                    count_ions = self.statistics_dict['count_ion_present']
                    for elmt, oxi_state, occ in site_species:
                        if elmt not in ion_stat:
                            ion_stat[elmt] = {}
                            count_ions[elmt] = {}
                        if oxi_state not in ion_stat[elmt]:
                            ion_stat[elmt][oxi_state] = {}
                            count_ions[elmt][oxi_state] = 0.0
                        count_ions[elmt][oxi_state] += occ
                        for ce_symbol, fraction in site_envs:
                            if fraction is None:
                                continue
                            if ce_symbol not in ion_stat[elmt][oxi_state]:
                                ion_stat[elmt][oxi_state][ce_symbol] = 0.0
                            ion_stat[elmt][oxi_state][ce_symbol] += occ * fraction
                            if ce_symbol not in ce_ion_stat:
                                ce_ion_stat[ce_symbol] = {}
                            if elmt not in ce_ion_stat[ce_symbol]:
                                ce_ion_stat[ce_symbol][elmt] = {}
                            if oxi_state not in ce_ion_stat[ce_symbol][elmt]:
                                ce_ion_stat[ce_symbol][elmt][oxi_state] = 0.0
                            ce_ion_stat[ce_symbol][elmt][oxi_state] += occ * fraction
        self.statistics_dict['anion_number'] = len(self.statistics_dict['anion_list'])
        self.statistics_dict['anion_atom_number'] = len(self.statistics_dict['anion_atom_list'])
        self.statistics_dict['cation_number'] = len(self.statistics_dict['cation_list'])
        self.statistics_dict['cation_atom_number'] = len(self.statistics_dict['cation_atom_list'])
        self.statistics_dict['neutral_number'] = len(self.statistics_dict['neutral_list'])
        self.statistics_dict['neutral_atom_number'] = len(self.statistics_dict['neutral_atom_list'])
        for elmt, envs in atom_stat.items():
            sumelement = count_atoms[elmt]
            fraction_atom_stat[elmt] = {env: fraction / sumelement for env, fraction in envs.items()}
        for ce_symbol, atoms in ce_atom_stat.items():
            sumsymbol = count_ce[ce_symbol]
            fraction_ce_atom_stat[ce_symbol] = {atom: fraction / sumsymbol for atom, fraction in atoms.items()}
        ion_stat = self.statistics_dict['ion_coordination_environments_present']
        fraction_ion_stat = self.statistics_dict['fraction_ion_coordination_environments_present']
        ce_ion_stat = self.statistics_dict['coordination_environments_ion_present']
        fraction_ce_ion_stat = self.statistics_dict['fraction_coordination_environments_ion_present']
        count_ions = self.statistics_dict['count_ion_present']
        for elmt, oxi_states_envs in ion_stat.items():
            fraction_ion_stat[elmt] = {}
            for oxi_state, envs in oxi_states_envs.items():
                sumspecie = count_ions[elmt][oxi_state]
                fraction_ion_stat[elmt][oxi_state] = {env: fraction / sumspecie for env, fraction in envs.items()}
        for ce_symbol, ions in ce_ion_stat.items():
            fraction_ce_ion_stat[ce_symbol] = {}
            sum_ce = np.sum([np.sum(list(oxi_states.values())) for elmt, oxi_states in ions.items()])
            for elmt, oxi_states in ions.items():
                fraction_ce_ion_stat[ce_symbol][elmt] = {oxistate: fraction / sum_ce for oxistate, fraction in oxi_states.items()}

    def get_site_info_for_specie_ce(self, specie, ce_symbol):
        """
        Get list of indices that have the given specie with a given Coordination environment.

        Args:
            specie: Species to get.
            ce_symbol: Symbol of the coordination environment to get.

        Returns:
            dict: Keys are 'isites', 'fractions', 'csms' which contain list of indices in the structure
                that have the given specie in the given environment, their fraction and continuous
                symmetry measures.
        """
        element = specie.symbol
        oxi_state = specie.oxi_state
        isites = []
        csms = []
        fractions = []
        for isite, site in enumerate(self.structure):
            if element in [sp.symbol for sp in site.species] and (self.valences == 'undefined' or oxi_state == self.valences[isite]):
                for ce_dict in self.coordination_environments[isite]:
                    if ce_symbol == ce_dict['ce_symbol']:
                        isites.append(isite)
                        csms.append(ce_dict['csm'])
                        fractions.append(ce_dict['ce_fraction'])
        return {'isites': isites, 'fractions': fractions, 'csms': csms}

    def get_site_info_for_specie_allces(self, specie, min_fraction=0):
        """
        Get list of indices that have the given specie.

        Args:
            specie: Species to get.
            min_fraction: Minimum fraction of the coordination environment.

        Returns:
            dict: with the list of coordination environments for the given species, the indices of the sites
                in which they appear, their fractions and continuous symmetry measures.
        """
        allces = {}
        element = specie.symbol
        oxi_state = specie.oxi_state
        for isite, site in enumerate(self.structure):
            if element in [sp.symbol for sp in site.species] and self.valences == 'undefined' or oxi_state == self.valences[isite]:
                if self.coordination_environments[isite] is None:
                    continue
                for ce_dict in self.coordination_environments[isite]:
                    if ce_dict['ce_fraction'] < min_fraction:
                        continue
                    if ce_dict['ce_symbol'] not in allces:
                        allces[ce_dict['ce_symbol']] = {'isites': [], 'fractions': [], 'csms': []}
                    allces[ce_dict['ce_symbol']]['isites'].append(isite)
                    allces[ce_dict['ce_symbol']]['fractions'].append(ce_dict['ce_fraction'])
                    allces[ce_dict['ce_symbol']]['csms'].append(ce_dict['csm'])
        return allces

    def get_statistics(self, statistics_fields=DEFAULT_STATISTICS_FIELDS, bson_compatible=False):
        """
        Get the statistics of environments for this structure.

        Args:
            statistics_fields: Which statistics to get.
            bson_compatible: Whether to make the dictionary BSON-compatible.

        Returns:
            dict: with the requested statistics.
        """
        if self.statistics_dict is None:
            self.setup_statistic_lists()
        if statistics_fields == 'ALL':
            statistics_fields = list(self.statistics_dict)
        if bson_compatible:
            dd = jsanitize({field: self.statistics_dict[field] for field in statistics_fields})
        else:
            dd = {field: self.statistics_dict[field] for field in statistics_fields}
        return dd

    def contains_only_one_anion_atom(self, anion_atom):
        """
        Whether this LightStructureEnvironments concerns a structure with only one given anion atom type.

        Args:
            anion_atom: Anion (e.g. O, ...). The structure could contain O2- and O- though.

        Returns:
            bool: True if this LightStructureEnvironments concerns a structure with only one given anion_atom.
        """
        return len(self.statistics_dict['anion_atom_list']) == 1 and anion_atom in self.statistics_dict['anion_atom_list']

    def contains_only_one_anion(self, anion):
        """
        Whether this LightStructureEnvironments concerns a structure with only one given anion type.

        Args:
            anion: Anion (e.g. O2-, ...).

        Returns:
            bool: True if this LightStructureEnvironments concerns a structure with only one given anion.
        """
        return len(self.statistics_dict['anion_list']) == 1 and anion in self.statistics_dict['anion_list']

    def site_contains_environment(self, isite, ce_symbol):
        """
        Whether a given site contains a given coordination environment.

        Args:
            isite: Index of the site.
            ce_symbol: Symbol of the coordination environment.

        Returns:
            bool: True if the site contains the given coordination environment.
        """
        if self.coordination_environments[isite] is None:
            return False
        return ce_symbol in [ce_dict['ce_symbol'] for ce_dict in self.coordination_environments[isite]]

    def site_has_clear_environment(self, isite, conditions=None):
        """
        Whether a given site has a "clear" environments.

        A "clear" environment is somewhat arbitrary. You can pass (multiple) conditions, e.g. the environment should
        have a continuous symmetry measure lower than this, a fraction higher than that, ...

        Args:
            isite: Index of the site.
            conditions: Conditions to be checked for an environment to be "clear".

        Returns:
            bool: True if the site has a clear environment.
        """
        if self.coordination_environments[isite] is None:
            raise ValueError(f'Coordination environments have not been determined for site {isite}')
        if conditions is None:
            return len(self.coordination_environments[isite]) == 1
        ce = max(self.coordination_environments[isite], key=lambda x: x['ce_fraction'])
        for condition in conditions:
            target = condition['target']
            if target == 'ce_fraction':
                if ce[target] < condition['minvalue']:
                    return False
            elif target == 'csm':
                if ce[target] > condition['maxvalue']:
                    return False
            elif target == 'number_of_ces':
                if ce[target] > condition['maxnumber']:
                    return False
            else:
                raise ValueError(f'Target {target!r} for condition of clear environment is not allowed')
        return True

    def structure_has_clear_environments(self, conditions=None, skip_none=True, skip_empty=False):
        """
        Whether all sites in a structure have "clear" environments.

        Args:
            conditions: Conditions to be checked for an environment to be "clear".
            skip_none: Whether to skip sites for which no environments have been computed.
            skip_empty: Whether to skip sites for which no environments could be found.

        Returns:
            bool: True if all the sites in the structure have clear environments.
        """
        for isite in range(len(self.structure)):
            if self.coordination_environments[isite] is None:
                if skip_none:
                    continue
                return False
            if len(self.coordination_environments[isite]) == 0:
                if skip_empty:
                    continue
                return False
            if not self.site_has_clear_environment(isite=isite, conditions=conditions):
                return False
        return True

    def clear_environments(self, conditions=None):
        """
        Get the clear environments in the structure.

        Args:
            conditions: Conditions to be checked for an environment to be "clear".

        Returns:
            list: Clear environments in this structure.
        """
        clear_envs_list = set()
        for isite in range(len(self.structure)):
            if self.coordination_environments[isite] is None:
                continue
            if len(self.coordination_environments[isite]) == 0:
                continue
            if self.site_has_clear_environment(isite=isite, conditions=conditions):
                ce = max(self.coordination_environments[isite], key=lambda x: x['ce_fraction'])
                clear_envs_list.add(ce['ce_symbol'])
        return list(clear_envs_list)

    def structure_contains_atom_environment(self, atom_symbol, ce_symbol):
        """
        Checks whether the structure contains a given atom in a given environment.

        Args:
            atom_symbol: Symbol of the atom.
            ce_symbol: Symbol of the coordination environment.

        Returns:
            bool: True if the coordination environment is found for the given atom.
        """
        for isite, site in enumerate(self.structure):
            if Element(atom_symbol) in site.species.element_composition and self.site_contains_environment(isite, ce_symbol):
                return True
        return False

    def environments_identified(self):
        """
        Return the set of environments identified in this structure.

        Returns:
            set: environments identified in this structure.
        """
        return {ce['ce_symbol'] for celist in self.coordination_environments if celist is not None for ce in celist}

    @property
    def uniquely_determines_coordination_environments(self):
        """True if the coordination environments are uniquely determined."""
        return self.strategy.uniquely_determines_coordination_environments

    def __eq__(self, other: object) -> bool:
        """
        Equality method that checks if the LightStructureEnvironments object is equal to another
        LightStructureEnvironments object. Two LightStructureEnvironments objects are equal if the strategy used
        is the same, if the structure is the same, if the valences used in the strategies are the same, if the
        coordination environments and the neighbors determined by the strategy are the same.

        Args:
            other: LightStructureEnvironments object to compare with.

        Returns:
            bool: True if both objects are equal.
        """
        if not isinstance(other, LightStructureEnvironments):
            return NotImplemented
        is_equal = self.strategy == other.strategy and self.structure == other.structure and (self.coordination_environments == other.coordination_environments) and (self.valences == other.valences) and (self.neighbors_sets == other.neighbors_sets)
        this_sites = [ss['site'] for ss in self._all_nbs_sites]
        other_sites = [ss['site'] for ss in other._all_nbs_sites]
        this_indices = [ss['index'] for ss in self._all_nbs_sites]
        other_indices = [ss['index'] for ss in other._all_nbs_sites]
        return is_equal and this_sites == other_sites and (this_indices == other_indices)

    def as_dict(self):
        """
        Returns:
            dict: Bson-serializable representation of the LightStructureEnvironments object.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'strategy': self.strategy.as_dict(), 'structure': self.structure.as_dict(), 'coordination_environments': self.coordination_environments, 'all_nbs_sites': [{'site': PeriodicSite(species=nb_site['site'].species, coords=nb_site['site'].frac_coords, lattice=nb_site['site'].lattice, to_unit_cell=False, coords_are_cartesian=False, properties=nb_site['site'].properties).as_dict(), 'index': nb_site['index'], 'image_cell': [int(ii) for ii in nb_site['image_cell']]} for nb_site in self._all_nbs_sites], 'neighbors_sets': [[nb_set.as_dict() for nb_set in site_nb_sets] if site_nb_sets is not None else None for site_nb_sets in self.neighbors_sets], 'valences': self.valences}

    @classmethod
    def from_dict(cls, dct) -> Self:
        """
        Reconstructs the LightStructureEnvironments object from a dict representation of the
        LightStructureEnvironments created using the as_dict method.

        Args:
            dct: dict representation of the LightStructureEnvironments object.

        Returns:
            LightStructureEnvironments object.
        """
        structure = MontyDecoder().process_decoded(dct['structure'])
        all_nbs_sites = []
        for nb_site in dct['all_nbs_sites']:
            periodic_site = MontyDecoder().process_decoded(nb_site['site'])
            site = PeriodicNeighbor(species=periodic_site.species, coords=periodic_site.frac_coords, lattice=periodic_site.lattice, properties=periodic_site.properties)
            if 'image_cell' in nb_site:
                image_cell = np.array(nb_site['image_cell'], int)
            else:
                diff = site.frac_coords - structure[nb_site['index']].frac_coords
                rounddiff = np.round(diff)
                if not np.allclose(diff, rounddiff):
                    raise ValueError('Weird, differences between one site in a periodic image cell is not integer ...')
                image_cell = np.array(rounddiff, int)
            all_nbs_sites.append({'site': site, 'index': nb_site['index'], 'image_cell': image_cell})
        neighbors_sets = [[cls.NeighborsSet.from_dict(nb_set, structure=structure, all_nbs_sites=all_nbs_sites) for nb_set in site_nb_sets] if site_nb_sets is not None else None for site_nb_sets in dct['neighbors_sets']]
        return cls(strategy=MontyDecoder().process_decoded(dct['strategy']), coordination_environments=dct['coordination_environments'], all_nbs_sites=all_nbs_sites, neighbors_sets=neighbors_sets, structure=structure, valences=dct['valences'])