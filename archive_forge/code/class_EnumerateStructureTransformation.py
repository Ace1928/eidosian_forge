from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
class EnumerateStructureTransformation(AbstractTransformation):
    """Order a disordered structure using enumlib. For complete orderings, this
    generally produces fewer structures that the OrderDisorderedStructure
    transformation, and at a much faster speed.
    """

    def __init__(self, min_cell_size: int=1, max_cell_size: int=1, symm_prec: float=0.1, refine_structure: bool=False, enum_precision_parameter: float=0.001, check_ordered_symmetry: bool=True, max_disordered_sites: int | None=None, sort_criteria: str | Callable='ewald', timeout: float | None=None, n_jobs: int=-1):
        """
        Args:
            min_cell_size:
                The minimum cell size wanted. Must be an int. Defaults to 1.
            max_cell_size:
                The maximum cell size wanted. Must be an int. Defaults to 1.
            symm_prec:
                Tolerance to use for symmetry.
            refine_structure:
                This parameter has the same meaning as in enumlib_caller.
                If you are starting from a structure that has been relaxed via
                some electronic structure code, it is usually much better to
                start with symmetry determination and then obtain a refined
                structure. The refined structure have cell parameters and
                atomic positions shifted to the expected symmetry positions,
                which makes it much less sensitive precision issues in enumlib.
                If you are already starting from an experimental cif, refinement
                should have already been done and it is not necessary. Defaults
                to False.
            enum_precision_parameter (float): Finite precision parameter for
                enumlib. Default of 0.001 is usually ok, but you might need to
                tweak it for certain cells.
            check_ordered_symmetry (bool): Whether to check the symmetry of
                the ordered sites. If the symmetry of the ordered sites is
                lower, the lowest symmetry ordered sites is included in the
                enumeration. This is important if the ordered sites break
                symmetry in a way that is important getting possible
                structures. But sometimes including ordered sites
                slows down enumeration to the point that it cannot be
                completed. Switch to False in those cases. Defaults to True.
            max_disordered_sites (int):
                An alternate parameter to max_cell size. Will sequentially try
                larger and larger cell sizes until (i) getting a result or (ii)
                the number of disordered sites in the cell exceeds
                max_disordered_sites. Must set max_cell_size to None when using
                this parameter.
            sort_criteria (str or callable): Sort by Ewald energy ("ewald", must have oxidation states and slow) or
                M3GNet relaxed energy ("m3gnet_relax", which is the most accurate but most expensive and provides
                pre-relaxed structures - needs m3gnet package installed) or by M3GNet static energy ("m3gnet_static")
                or by number of sites ("nsites", the fastest, the default). The expense of m3gnet_relax or m3gnet_static
                can be worth it if it significantly reduces the number of structures to be considered. m3gnet_relax
                speeds up the subsequent DFT calculations. Alternatively, a callable can be supplied that returns a
                (Structure, energy) tuple.
            timeout (float): timeout in minutes to pass to EnumlibAdaptor.
            n_jobs (int): Number of parallel jobs used to compute energy criteria. This is used only when the Ewald
                or m3gnet or callable sort_criteria is used. Default is -1, which uses all available CPUs.
        """
        self.symm_prec = symm_prec
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.refine_structure = refine_structure
        self.enum_precision_parameter = enum_precision_parameter
        self.check_ordered_symmetry = check_ordered_symmetry
        self.max_disordered_sites = max_disordered_sites
        self.sort_criteria = sort_criteria
        self.timeout = timeout
        self.n_jobs = n_jobs
        if max_cell_size and max_disordered_sites:
            raise ValueError('Cannot set both max_cell_size and max_disordered_sites!')

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False) -> Structure | list[dict]:
        """Returns either a single ordered structure or a sequence of all ordered
        structures.

        Args:
            structure: Structure to order.
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Depending on returned_ranked list, either a transformed structure
            or a list of dictionaries, where each dictionary is of the form
            {"structure" = .... , "other_arguments"}

            The list of ordered structures is ranked by Ewald energy / atom, if
            the input structure is an oxidation state decorated structure.
            Otherwise, it is ranked by number of sites, with smallest number of
            sites first.
        """
        try:
            num_to_return = int(return_ranked_list)
        except ValueError:
            num_to_return = 1
        if self.refine_structure:
            finder = SpacegroupAnalyzer(structure, self.symm_prec)
            structure = finder.get_refined_structure()
        contains_oxidation_state = all((hasattr(sp, 'oxi_state') and sp.oxi_state != 0 for sp in structure.elements))
        structures = None
        if structure.is_ordered:
            warnings.warn(f'Enumeration skipped for structure with composition {structure.composition} because it is ordered')
            structures = [structure.copy()]
        if self.max_disordered_sites:
            n_disordered = sum((1 for site in structure if not site.is_ordered))
            if n_disordered > self.max_disordered_sites:
                raise ValueError(f'Too many disordered sites! ({n_disordered} > {self.max_disordered_sites})')
            max_cell_sizes: Iterable[int] = range(self.min_cell_size, int(math.floor(self.max_disordered_sites / n_disordered)) + 1)
        else:
            max_cell_sizes = [self.max_cell_size]
        for max_cell_size in max_cell_sizes:
            adaptor = EnumlibAdaptor(structure, min_cell_size=self.min_cell_size, max_cell_size=max_cell_size, symm_prec=self.symm_prec, refine_structure=False, enum_precision_parameter=self.enum_precision_parameter, check_ordered_symmetry=self.check_ordered_symmetry, timeout=self.timeout)
            try:
                adaptor.run()
                structures = adaptor.structures
                if structures:
                    break
            except EnumError:
                warnings.warn(f'Unable to enumerate for max_cell_size = {max_cell_size!r}')
        if structures is None:
            raise ValueError('Unable to enumerate')
        original_latt = structure.lattice
        inv_latt = np.linalg.inv(original_latt.matrix)
        ewald_matrices = {}
        if not callable(self.sort_criteria) and self.sort_criteria.startswith('m3gnet'):
            import matgl
            from matgl.ext.ase import M3GNetCalculator, Relaxer
            if self.sort_criteria == 'm3gnet_relax':
                potential = matgl.load_model('M3GNet-MP-2021.2.8-PES')
                m3gnet_model = Relaxer(potential=potential)
            elif self.sort_criteria == 'm3gnet':
                potential = matgl.load_model('M3GNet-MP-2021.2.8-PES')
                m3gnet_model = M3GNetCalculator(potential=potential, stress_weight=0.01)

        def _get_stats(struct):
            if callable(self.sort_criteria):
                struct, energy = self.sort_criteria(struct)
                return {'num_sites': len(struct), 'energy': energy, 'structure': struct}
            if contains_oxidation_state and self.sort_criteria == 'ewald':
                new_latt = struct.lattice
                transformation = np.dot(new_latt.matrix, inv_latt)
                transformation = tuple((tuple((int(round(cell)) for cell in row)) for row in transformation))
                if transformation not in ewald_matrices:
                    s_supercell = structure * transformation
                    ewald = EwaldSummation(s_supercell)
                    ewald_matrices[transformation] = ewald
                else:
                    ewald = ewald_matrices[transformation]
                energy = ewald.compute_sub_structure(struct)
                return {'num_sites': len(struct), 'energy': energy, 'structure': struct}
            if self.sort_criteria.startswith('m3gnet'):
                if self.sort_criteria == 'm3gnet_relax':
                    relax_results = m3gnet_model.relax(struct)
                    energy = float(relax_results['trajectory'].energies[-1])
                    struct = relax_results['final_structure']
                else:
                    from pymatgen.io.ase import AseAtomsAdaptor
                    atoms = AseAtomsAdaptor().get_atoms(struct)
                    m3gnet_model.calculate(atoms)
                    energy = float(m3gnet_model.results['energy'])
                return {'num_sites': len(struct), 'energy': energy, 'structure': struct}
            return {'num_sites': len(struct), 'structure': struct}
        all_structures = Parallel(n_jobs=self.n_jobs)((delayed(_get_stats)(struct) for struct in structures))

        def sort_func(s):
            return s['energy'] / s['num_sites'] if callable(self.sort_criteria) or self.sort_criteria.startswith('m3gnet') or (contains_oxidation_state and self.sort_criteria == 'ewald') else s['num_sites']
        self._all_structures = sorted(all_structures, key=sort_func)
        if return_ranked_list:
            return self._all_structures[0:num_to_return]
        return self._all_structures[0]['structure']

    def __repr__(self):
        return 'EnumerateStructureTransformation'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True