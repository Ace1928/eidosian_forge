from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class AbstractChemenvStrategy(MSONable, abc.ABC):
    """
    Class used to define a Chemenv strategy for the neighbors and coordination environment to be applied to a
    StructureEnvironments object.
    """
    AC = AdditionalConditions()
    STRATEGY_OPTIONS: ClassVar[dict[str, dict]] = dict()
    STRATEGY_DESCRIPTION: str | None = None
    STRATEGY_INFO_FIELDS: ClassVar[list] = []
    DEFAULT_SYMMETRY_MEASURE_TYPE = 'csm_wcs_ctwcc'

    def __init__(self, structure_environments=None, symmetry_measure_type=DEFAULT_SYMMETRY_MEASURE_TYPE):
        """
        Abstract constructor for the all chemenv strategies.

        Args:
            structure_environments: StructureEnvironments object containing all the information on the
                coordination of the sites in a structure.
        """
        self.structure_environments = None
        if structure_environments is not None:
            self.set_structure_environments(structure_environments)
        self._symmetry_measure_type = symmetry_measure_type

    @property
    def symmetry_measure_type(self):
        """Type of symmetry measure."""
        return self._symmetry_measure_type

    def set_structure_environments(self, structure_environments):
        """Set the structure environments to this strategy.

        Args:
            structure_environments: StructureEnvironments object.
        """
        self.structure_environments = structure_environments
        if not isinstance(self.structure_environments.voronoi, DetailedVoronoiContainer):
            raise ValueError('Voronoi Container not of type "DetailedVoronoiContainer"')
        self.prepare_symmetries()

    def prepare_symmetries(self):
        """Prepare the symmetries for the structure contained in the structure environments."""
        try:
            self.spg_analyzer = SpacegroupAnalyzer(self.structure_environments.structure)
            self.symops = self.spg_analyzer.get_symmetry_operations()
        except Exception:
            self.symops = []

    def equivalent_site_index_and_transform(self, psite):
        """Get the equivalent site and corresponding symmetry+translation transformations.

        Args:
            psite: Periodic site.

        Returns:
            Equivalent site in the unit cell, translations and symmetry transformation.
        """
        try:
            isite = self.structure_environments.structure.index(psite)
        except ValueError:
            try:
                uc_psite = psite.to_unit_cell()
                isite = self.structure_environments.structure.index(uc_psite)
            except ValueError:
                for isite2, site2 in enumerate(self.structure_environments.structure):
                    if psite.is_periodic_image(site2):
                        isite = isite2
                        break
        this_site = self.structure_environments.structure[isite]
        dthis_site = psite.frac_coords - this_site.frac_coords
        equiv_site = self.structure_environments.structure[self.structure_environments.sites_map[isite]].to_unit_cell()
        dequivsite = self.structure_environments.structure[self.structure_environments.sites_map[isite]].frac_coords - equiv_site.frac_coords
        found = False
        tolerances = [1e-08, 1e-07, 1e-06, 1e-05, 0.0001]
        for tolerance in tolerances:
            for sym_op in self.symops:
                new_site = PeriodicSite(equiv_site._species, sym_op.operate(equiv_site.frac_coords), equiv_site._lattice)
                if new_site.is_periodic_image(this_site, tolerance=tolerance):
                    sym_trafo = sym_op
                    d_this_site2 = this_site.frac_coords - new_site.frac_coords
                    found = True
                    break
            if not found:
                sym_ops = [SymmOp.from_rotation_and_translation()]
                for sym_op in sym_ops:
                    new_site = PeriodicSite(equiv_site._species, sym_op.operate(equiv_site.frac_coords), equiv_site._lattice)
                    if new_site.is_periodic_image(this_site, tolerance=tolerance):
                        sym_trafo = sym_op
                        d_this_site2 = this_site.frac_coords - new_site.frac_coords
                        found = True
                        break
            if found:
                break
        if not found:
            raise EquivalentSiteSearchError(psite)
        return (self.structure_environments.sites_map[isite], dequivsite, dthis_site + d_this_site2, sym_trafo)

    @abc.abstractmethod
    def get_site_neighbors(self, site):
        """
        Applies the strategy to the structure_environments object in order to get the neighbors of a given site.

        Args:
            site: Site for which the neighbors are looked for
            structure_environments: StructureEnvironments object containing all the information needed to get the
                neighbors of the site

        Returns:
            The list of neighbors of the site. For complex strategies, where one allows multiple solutions, this
            can return a list of list of neighbors.
        """
        raise NotImplementedError

    @property
    def uniquely_determines_coordination_environments(self):
        """Returns True if the strategy leads to a unique coordination environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_site_coordination_environment(self, site):
        """
        Applies the strategy to the structure_environments object in order to define the coordination environment of
        a given site.

        Args:
            site: Site for which the coordination environment is looked for

        Returns:
            The coordination environment of the site. For complex strategies, where one allows multiple
            solutions, this can return a list of coordination environments for the site.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_site_coordination_environments(self, site):
        """
        Applies the strategy to the structure_environments object in order to define the coordination environment of
        a given site.

        Args:
            site: Site for which the coordination environment is looked for

        Returns:
            The coordination environment of the site. For complex strategies, where one allows multiple
            solutions, this can return a list of coordination environments for the site.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_site_coordination_environments_fractions(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None, ordered=True, min_fraction=0, return_maps=True, return_strategy_dict_info=False):
        """
        Applies the strategy to the structure_environments object in order to define the coordination environment of
        a given site.

        Args:
            site: Site for which the coordination environment is looked for

        Returns:
            The coordination environment of the site. For complex strategies, where one allows multiple
            solutions, this can return a list of coordination environments for the site.
        """
        raise NotImplementedError

    def get_site_ce_fractions_and_neighbors(self, site, full_ce_info=False, strategy_info=False):
        """
        Applies the strategy to the structure_environments object in order to get coordination environments, their
        fraction, csm, geometry_info, and neighbors

        Args:
            site: Site for which the above information is sought

        Returns:
            The list of neighbors of the site. For complex strategies, where one allows multiple solutions, this
        can return a list of list of neighbors.
        """
        isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
        geoms_and_maps_list = self.get_site_coordination_environments_fractions(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_maps=True, return_strategy_dict_info=True)
        if geoms_and_maps_list is None:
            return None
        site_nbs_sets = self.structure_environments.neighbors_sets[isite]
        ce_and_neighbors = []
        for fractions_dict in geoms_and_maps_list:
            ce_map = fractions_dict['ce_map']
            ce_nb_set = site_nbs_sets[ce_map[0]][ce_map[1]]
            neighbors = [{'site': nb_site_and_index['site'], 'index': nb_site_and_index['index']} for nb_site_and_index in ce_nb_set.neighb_sites_and_indices]
            fractions_dict['neighbors'] = neighbors
            ce_and_neighbors.append(fractions_dict)
        return ce_and_neighbors

    def set_option(self, option_name, option_value):
        """Set up a given option for this strategy.

        Args:
            option_name: Name of the option.
            option_value: Value for this option.
        """
        setattr(self, option_name, option_value)

    def setup_options(self, all_options_dict):
        """Set up options for this strategy based on a dict.

        Args:
            all_options_dict: Dict of option_name->option_value.
        """
        for option_name, option_value in all_options_dict.items():
            self.set_option(option_name, option_value)

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Equality method that should be implemented for any strategy

        Args:
            other: strategy to be compared with the current one
        """
        raise NotImplementedError

    def __str__(self):
        out = f'  Chemenv Strategy {type(self).__name__!r}\n'
        out += f'  {'=' * (19 + len(type(self).__name__))}\n\n'
        out += f'  Description :\n  {'-' * 13}\n'
        out += self.STRATEGY_DESCRIPTION
        out += '\n\n'
        out += f'  Options :\n  {'-' * 9}\n'
        for option_name in self.STRATEGY_OPTIONS:
            out += f'   - {option_name} : {getattr(self, option_name)}\n'
        return out

    @abc.abstractmethod
    def as_dict(self):
        """
        Bson-serializable dict representation of the SimplestChemenvStrategy object.

        Returns:
            Bson-serializable dict representation of the SimplestChemenvStrategy object.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct) -> Self:
        """
        Reconstructs the SimpleAbundanceChemenvStrategy object from a dict representation of the
        SimpleAbundanceChemenvStrategy object created using the as_dict method.

        Args:
            dct: dict representation of the SimpleAbundanceChemenvStrategy object

        Returns:
            StructureEnvironments object.
        """
        raise NotImplementedError