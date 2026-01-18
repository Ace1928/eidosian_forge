from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
class SiteCollection(collections.abc.Sequence, ABC):
    """Basic SiteCollection. Essentially a sequence of Sites or PeriodicSites.
    This serves as a base class for Molecule (a collection of Site, i.e., no
    periodicity) and Structure (a collection of PeriodicSites, i.e.,
    periodicity). Not meant to be instantiated directly.
    """
    DISTANCE_TOLERANCE = 0.5
    _properties: dict

    @property
    def sites(self) -> list[Site]:
        """Returns an iterator for the sites in the Structure."""
        return self._sites

    @sites.setter
    def sites(self, sites: Sequence[PeriodicSite]) -> None:
        """Sets the sites in the Structure."""
        is_mutable = isinstance(self._sites, list)
        self._sites = list(sites) if is_mutable else tuple(sites)

    @abstractmethod
    def copy(self) -> SiteCollection:
        """Returns a copy of itself. Concrete subclasses should implement this
        method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_distance(self, i: int, j: int) -> float:
        """Returns distance between sites at index i and j.

        Args:
            i: 1st site index
            j: 2nd site index

        Returns:
            Distance between sites at index i and index j.
        """
        raise NotImplementedError

    @property
    def distance_matrix(self) -> np.ndarray:
        """Returns the distance matrix between all sites in the structure. For
        periodic structures, this is overwritten to return the nearest image
        distance.
        """
        return all_distances(self.cart_coords, self.cart_coords)

    @property
    def species(self) -> list[Element | Species]:
        """Only works for ordered structures.

        Raises:
            AttributeError: If structure is disordered.

        Returns:
            list[Species]: species at each site of the structure.
        """
        if not self.is_ordered:
            raise AttributeError('species property only supports ordered structures!')
        return [site.specie for site in self]

    @property
    def species_and_occu(self) -> list[Composition]:
        """List of species and occupancies at each site of the structure."""
        return [site.species for site in self]

    @property
    @deprecated(message='Use n_type_sp instead.')
    def ntypesp(self) -> int:
        """Number of types of atoms."""
        return len(self.types_of_species)

    @property
    def n_elems(self) -> int:
        """Number of types of atoms."""
        return len(self.types_of_species)

    @property
    def types_of_species(self) -> tuple[Element | Species | DummySpecies]:
        """List of types of specie."""
        types: list[Element | Species | DummySpecies] = []
        for site in self:
            for sp, amt in site.species.items():
                if amt != 0:
                    types.append(sp)
        return tuple(sorted(set(types)))

    @property
    def types_of_specie(self) -> tuple[Element | Species | DummySpecies]:
        """Specie->Species rename. Maintained for backwards compatibility."""
        return self.types_of_species

    def group_by_types(self) -> Iterator[Site | PeriodicSite]:
        """Iterate over species grouped by type."""
        for sp_typ in self.types_of_species:
            for site in self:
                if site.specie == sp_typ:
                    yield site

    def indices_from_symbol(self, symbol: str) -> tuple[int, ...]:
        """Returns a tuple with the sequential indices of the sites
        that contain an element with the given chemical symbol.
        """
        return tuple((idx for idx, specie in enumerate(self.species) if specie.symbol == symbol))

    @property
    def symbol_set(self) -> tuple[str, ...]:
        """Tuple with the set of chemical symbols.
        Note that len(symbol_set) == len(types_of_specie).
        """
        return tuple(sorted((specie.symbol for specie in self.types_of_species)))

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        """List of atomic numbers."""
        try:
            return tuple((site.specie.Z for site in self))
        except AttributeError:
            raise AttributeError('atomic_numbers available only for ordered Structures')

    @property
    def site_properties(self) -> dict[str, Sequence]:
        """The site properties as a dict of sequences.
        E.g. {"magmom": (5, -5), "charge": (-4, 4)}.
        """
        prop_keys: set[str] = set()
        for site in self:
            prop_keys.update(site.properties)
        return {key: [site.properties.get(key) for site in self] for key in prop_keys}

    @property
    def labels(self) -> list[str]:
        """Site labels as a list."""
        return [site.label for site in self]

    def __contains__(self, site: object) -> bool:
        return site in self.sites

    def __iter__(self) -> Iterator[Site]:
        return iter(self.sites)

    def __getitem__(self, ind: int | slice) -> Site:
        return self.sites[ind]

    def __len__(self) -> int:
        return len(self.sites)

    def __hash__(self) -> int:
        return hash(self.composition)

    @property
    def num_sites(self) -> int:
        """Number of sites."""
        return len(self)

    @property
    def cart_coords(self) -> np.ndarray:
        """Returns an np.array of the Cartesian coordinates of sites in the structure."""
        return np.array([site.coords for site in self])

    @property
    def formula(self) -> str:
        """Returns the formula as a string."""
        return self.composition.formula

    @property
    def alphabetical_formula(self) -> str:
        """Returns the formula as a string."""
        return self.composition.alphabetical_formula

    @property
    def reduced_formula(self) -> str:
        """Returns the reduced formula as a string."""
        return self.composition.reduced_formula

    @property
    def elements(self) -> list[Element | Species | DummySpecies]:
        """Returns the elements in the structure as a list of Element objects."""
        return self.composition.elements

    @property
    def composition(self) -> Composition:
        """Returns the structure's corresponding Composition object."""
        elem_map: dict[Species, float] = collections.defaultdict(float)
        for site in self:
            for species, occu in site.species.items():
                elem_map[species] += occu
        return Composition(elem_map)

    @property
    def charge(self) -> float:
        """Returns the net charge of the structure based on oxidation states. If
        Elements are found, a charge of 0 is assumed.
        """
        charge = 0
        for site in self:
            for specie, amt in site.species.items():
                charge += (getattr(specie, 'oxi_state', 0) or 0) * amt
        return charge

    @property
    def is_ordered(self) -> bool:
        """Checks if structure is ordered, meaning no partial occupancies in any
        of the sites.
        """
        return all((site.is_ordered for site in self))

    def get_angle(self, i: int, j: int, k: int) -> float:
        """Returns angle specified by three sites.

        Args:
            i: 1st site index
            j: 2nd site index
            k: 3rd site index

        Returns:
            Angle in degrees.
        """
        vec_1 = self[i].coords - self[j].coords
        vec_2 = self[k].coords - self[j].coords
        return get_angle(vec_1, vec_2, units='degrees')

    def get_dihedral(self, i: int, j: int, k: int, l: int) -> float:
        """Returns dihedral angle specified by four sites.

        Args:
            i (int): 1st site index
            j (int): 2nd site index
            k (int): 3rd site index
            l (int): 4th site index

        Returns:
            Dihedral angle in degrees.
        """
        vec1 = self[k].coords - self[l].coords
        vec2 = self[j].coords - self[k].coords
        vec3 = self[i].coords - self[j].coords
        vec23 = np.cross(vec2, vec3)
        vec12 = np.cross(vec1, vec2)
        return math.degrees(math.atan2(np.linalg.norm(vec2) * np.dot(vec1, vec23), np.dot(vec12, vec23)))

    def is_valid(self, tol: float=DISTANCE_TOLERANCE) -> bool:
        """True if SiteCollection does not contain atoms that are too close
        together. Note that the distance definition is based on type of
        SiteCollection. Cartesian distances are used for non-periodic
        Molecules, while PBC is taken into account for periodic structures.

        Args:
            tol (float): Distance tolerance. Default is 0.5 Angstrom, which is fairly large.

        Returns:
            bool: True if SiteCollection does not contain atoms that are too close together.
        """
        if len(self) == 1:
            return True
        all_dists = self.distance_matrix[np.triu_indices(len(self), 1)]
        return np.min(all_dists) > tol

    @abstractmethod
    def to(self, filename: str='', fmt: FileFormats='') -> str | None:
        """Generates string representations (cif, json, poscar, ....) of SiteCollections (e.g.,
        molecules / structures). Should return str or None if written to a file.
        """
        raise NotImplementedError

    def to_file(self, filename: str='', fmt: FileFormats='') -> str | None:
        """A more intuitive alias for .to()."""
        return self.to(filename, fmt)

    @classmethod
    @abstractmethod
    def from_str(cls, input_string: str, fmt: Any) -> None:
        """Reads in SiteCollection from a string."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(cls, filename: str) -> None:
        """Reads in SiteCollection from a filename."""
        raise NotImplementedError

    def add_site_property(self, property_name: str, values: Sequence | np.ndarray) -> SiteCollection:
        """Adds a property to a site. Note: This is the preferred method
        for adding magnetic moments, selective dynamics, and related
        site-specific properties to a structure/molecule object.

        Examples:
            structure.add_site_property("magmom", [1.0, 0.0])
            structure.add_site_property("selective_dynamics", [[True, True, True], [False, False, False]])

        Args:
            property_name (str): The name of the property to add.
            values (list): A sequence of values. Must be same length as
                number of sites.

        Raises:
            ValueError: if len(values) != number of sites.

        Returns:
            SiteCollection: self with site property added.
        """
        if len(values) != len(self):
            raise ValueError(f'len(values)={len(values)!r} must equal sites in structure={len(self)}')
        for site, val in zip(self, values):
            site.properties[property_name] = val
        return self

    def remove_site_property(self, property_name: str) -> SiteCollection:
        """Removes a property to a site.

        Args:
            property_name (str): The name of the property to remove.

        Returns:
            SiteCollection: self with property removed.
        """
        for site in self:
            del site.properties[property_name]
        return self

    def replace_species(self, species_mapping: dict[SpeciesLike, SpeciesLike | dict[SpeciesLike, float]], in_place: bool=True) -> SiteCollection:
        """Swap species.

        Note that this clears the label of any affected site.

        Args:
            species_mapping (dict): Species to swap. Species can be elements too. E.g.,
                {Element("Li"): Element("Na")} performs a Li for Na substitution. The second species can
                be a sp_and_occu dict. For example, a site with 0.5 Si that is passed the mapping
                {Element('Si'): {Element('Ge'): 0.75, Element('C'): 0.25} } will have .375 Ge and .125 C.
            in_place (bool): Whether to perform the substitution in place or modify a copy.
                Defaults to True.

        Returns:
            SiteCollection: self or new SiteCollection (depending on in_place) with species replaced.
        """
        site_coll = self if in_place else self.copy()
        sp_mapping = {get_el_sp(k): v for k, v in species_mapping.items()}
        sp_to_replace = set(sp_mapping)
        sp_in_structure = set(self.composition)
        if not sp_in_structure >= sp_to_replace:
            warnings.warn(f'Some species to be substituted are not present in structure. Pls check your input. Species to be substituted = {sp_to_replace}; Species in structure = {sp_in_structure}')
        for site in site_coll:
            if sp_to_replace.intersection(site.species):
                comp = Composition()
                for sp, amt in site.species.items():
                    new_sp = sp_mapping.get(sp, sp)
                    try:
                        comp += Composition(new_sp) * amt
                    except Exception:
                        comp += {new_sp: amt}
                site.species = comp
                site.label = None
        return site_coll

    def add_oxidation_state_by_element(self, oxidation_states: dict[str, float]) -> SiteCollection:
        """Add oxidation states.

        Args:
            oxidation_states (dict): Dict of oxidation states.
                E.g., {"Li":1, "Fe":2, "P":5, "O":-2}

        Raises:
            ValueError if oxidation states are not specified for all elements.

        Returns:
            SiteCollection: self with oxidation states.
        """
        if (missing := ({el.symbol for el in self.composition} - {*oxidation_states})):
            raise ValueError(f'Oxidation states not specified for all elements, missing={missing!r}')
        for site in self:
            new_sp = {}
            for el, occu in site.species.items():
                new_sp[Species(el.symbol, oxidation_states[el.symbol])] = occu
            site.species = Composition(new_sp)
        return self

    def add_oxidation_state_by_site(self, oxidation_states: list[float]) -> SiteCollection:
        """Add oxidation states to a structure by site.

        Args:
            oxidation_states (list[float]): List of oxidation states.
                E.g. [1, 1, 1, 1, 2, 2, 2, 2, 5, 5, 5, 5, -2, -2, -2, -2]

        Raises:
            ValueError if oxidation states are not specified for all sites.

        Returns:
            SiteCollection: self with oxidation states.
        """
        if len(oxidation_states) != len(self):
            raise ValueError(f'Oxidation states of all sites must be specified, expected {len(self)} values, got {len(oxidation_states)}')
        for site, ox in zip(self, oxidation_states):
            new_sp = {}
            for el, occu in site.species.items():
                sym = el.symbol
                new_sp[Species(sym, ox)] = occu
            site.species = Composition(new_sp)
        return self

    def remove_oxidation_states(self) -> SiteCollection:
        """Removes oxidation states from a structure."""
        for site in self:
            new_sp: dict[Element, float] = collections.defaultdict(float)
            for el, occu in site.species.items():
                sym = el.symbol
                new_sp[Element(sym)] += occu
            site.species = Composition(new_sp)
        return self

    def add_oxidation_state_by_guess(self, **kwargs) -> SiteCollection:
        """Decorates the structure with oxidation state, guessing
        using Composition.oxi_state_guesses(). If multiple guesses are found
        we take the first one.

        Args:
            **kwargs: parameters to pass into oxi_state_guesses()
        """
        oxi_guess = self.composition.oxi_state_guesses(**kwargs)
        oxi_guess = oxi_guess or [{e.symbol: 0 for e in self.composition}]
        self.add_oxidation_state_by_element(oxi_guess[0])
        return self

    def add_spin_by_element(self, spins: dict[str, float]) -> SiteCollection:
        """Add spin states to structure.

        Args:
            spins (dict): Dict of spins associated with elements or species,
                e.g. {"Ni":+5} or {"Ni2+":5}
        """
        for site in self:
            new_species = {}
            for sp, occu in site.species.items():
                sym = sp.symbol
                oxi_state = getattr(sp, 'oxi_state', None)
                species = Species(sym, oxidation_state=oxi_state, spin=spins.get(str(sp), spins.get(sym)))
                new_species[species] = occu
            site.species = Composition(new_species)
        return self

    def add_spin_by_site(self, spins: Sequence[float]) -> SiteCollection:
        """Add spin states to structure by site.

        Args:
            spins (list): e.g. [+5, -5, 0, 0]
        """
        if len(spins) != len(self):
            raise ValueError(f'Spins for all sites must be specified, expected {len(self)} spins, got {len(spins)}')
        for site, spin in zip(self.sites, spins):
            new_species = {}
            for sp, occu in site.species.items():
                sym = sp.symbol
                oxi_state = getattr(sp, 'oxi_state', None)
                new_species[Species(sym, oxidation_state=oxi_state, spin=spin)] = occu
            site.species = Composition(new_species)
        return self

    def remove_spin(self) -> SiteCollection:
        """Remove spin states from structure."""
        for site in self:
            new_sp: dict[Element, float] = collections.defaultdict(float)
            for sp, occu in site.species.items():
                oxi_state = getattr(sp, 'oxi_state', None)
                new_sp[Species(sp.symbol, oxidation_state=oxi_state)] += occu
            site.species = Composition(new_sp)
        return self

    def extract_cluster(self, target_sites: list[Site], **kwargs) -> list[Site]:
        """Extracts a cluster of atoms based on bond lengths.

        Args:
            target_sites (list[Site]): Initial sites from which to nucleate cluster.
            **kwargs: kwargs passed through to CovalentBond.is_bonded.

        Returns:
            list[Site/PeriodicSite] Cluster of atoms.
        """
        cluster = list(target_sites)
        others = [site for site in self if site not in cluster]
        size = 0
        while len(cluster) > size:
            size = len(cluster)
            new_others = []
            for site in others:
                for site2 in cluster:
                    if CovalentBond.is_bonded(site, site2, **kwargs):
                        cluster.append(site)
                        break
                else:
                    new_others.append(site)
            others = new_others
        return cluster

    def _calculate(self, calculator: str | Calculator, verbose: bool=False) -> Calculator:
        """Performs an ASE calculation.

        Args:
            calculator (str | Calculator): An ASE Calculator or a string from the following case-insensitive
                options: "m3gnet", "gfn2-xtb", "chgnet".
            verbose (bool): whether to print stdout. Defaults to False.
                Has no effect when calculator=='chgnet'.

        Returns:
            Structure | Molecule: Structure or Molecule with new calc attribute containing
                result of ASE calculation.
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        if isinstance(self, Molecule) and isinstance(calculator, str) and (calculator.lower() in ('chgnet', 'm3gnet')):
            raise ValueError(f"Can't use calculator={calculator!r} for a Molecule.")
        calculator = self._prep_calculator(calculator)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(self)
        atoms.calc = calculator
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            atoms.get_potential_energy()
        return calculator

    def _relax(self, calculator: str | Calculator, relax_cell: bool=True, optimizer: str | Optimizer='FIRE', steps: int=500, fmax: float=0.1, stress_weight: float=0.01, opt_kwargs: dict | None=None, return_trajectory: bool=False, verbose: bool=False) -> Structure | Molecule | tuple[Structure | Molecule, TrajectoryObserver | Trajectory]:
        """Performs a structure relaxation using an ASE calculator.

        Args:
            calculator (str | ase.Calculator): An ASE Calculator or a string from the following options: "M3GNet",
                "gfn2-xtb".
            relax_cell (bool): whether to relax the lattice cell. Defaults to True.
            optimizer (str): name of the ASE optimizer class to use
            steps (int): max number of steps for relaxation. Defaults to 500.
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces. Defaults to 0.1.
            stress_weight (float): the stress weight for relaxation with M3GNet.
                Defaults to 0.01.
            opt_kwargs (dict): kwargs for the ASE optimizer class.
            return_trajectory (bool): Whether to return the trajectory of relaxation.
                Defaults to False.
            verbose (bool): whether to print stdout. Defaults to False.

        Returns:
            Structure | Molecule: Relaxed structure or molecule
        """
        from ase import optimize
        from ase.constraints import ExpCellFilter
        from ase.io import read
        from ase.optimize.optimize import Optimizer
        from pymatgen.io.ase import AseAtomsAdaptor
        opt_kwargs = opt_kwargs or {}
        is_molecule = isinstance(self, Molecule)
        run_uip = isinstance(calculator, str) and calculator.lower() in ('m3gnet', 'chgnet')
        calc_params = {} if is_molecule else dict(stress_weight=stress_weight)
        calculator = self._prep_calculator(calculator, **calc_params)

        def is_ase_optimizer(key):
            return isclass((obj := getattr(optimize, key))) and issubclass(obj, Optimizer)
        valid_keys = [key for key in dir(optimize) if is_ase_optimizer(key)]
        if isinstance(optimizer, str):
            if optimizer not in valid_keys:
                raise ValueError(f'Unknown optimizer={optimizer!r}, must be one of {valid_keys}')
            opt_class = getattr(optimize, optimizer)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(self)
        if return_trajectory:
            if run_uip:
                from matgl.ext.ase import TrajectoryObserver
                traj_observer = TrajectoryObserver(atoms)
            else:
                opt_kwargs.setdefault('trajectory', 'opt.traj')
        atoms.calc = calculator
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            if relax_cell:
                if is_molecule:
                    raise ValueError("Can't relax cell for a Molecule.")
                ecf = ExpCellFilter(atoms)
                dyn = opt_class(ecf, **opt_kwargs)
            else:
                dyn = opt_class(atoms, **opt_kwargs)
            dyn.run(fmax=fmax, steps=steps)
        system: Structure | Molecule = adaptor.get_molecule(atoms) if is_molecule else adaptor.get_structure(atoms)
        system.calc = atoms.calc
        system.dynamics = dyn.todict()
        if return_trajectory:
            if run_uip:
                traj_observer()
            else:
                traj_file = opt_kwargs['trajectory']
                traj_observer = read(traj_file, index=':')
            return (system, traj_observer)
        return system

    def _prep_calculator(self, calculator: Literal['m3gnet', 'gfn2-xtb'] | Calculator, **params) -> Calculator:
        """Convert string name of special ASE calculators into ASE calculator objects.

        Args:
            calculator: An ASE Calculator or a string from the following options: "m3gnet",
                "gfn2-xtb".
            **params: Parameters for the calculator.

        Returns:
            Calculator: ASE calculator object.
        """
        if inspect.isclass(calculator):
            return calculator(**params)
        if not isinstance(calculator, str):
            return calculator
        if calculator.lower() == 'chgnet':
            try:
                from chgnet.model import CHGNetCalculator
            except ImportError:
                raise ImportError('chgnet not installed. Try `pip install chgnet`.')
            return CHGNetCalculator()
        if calculator.lower() == 'm3gnet':
            try:
                import matgl
                from matgl.ext.ase import M3GNetCalculator
            except ImportError:
                raise ImportError('matgl not installed. Try `pip install matgl`.')
            potential = matgl.load_model('M3GNet-MP-2021.2.8-PES')
            return M3GNetCalculator(potential=potential, **params)
        if calculator.lower() == 'gfn2-xtb':
            try:
                from tblite.ase import TBLite
            except ImportError:
                raise ImportError('Must install tblite[ase]. Try `pip install tblite[ase]` (Linux)or `conda install -c conda-forge tblite-python` on (Mac/Windows).')
            return TBLite(method='GFN2-xTB', **params)
        raise ValueError(f'Unknown calculator={calculator!r}.')

    def to_ase_atoms(self, **kwargs) -> Atoms:
        """Converts the structure/molecule to an ase.Atoms object.

        Args:
            kwargs: Passed to ase.Atoms init.

        Returns:
            ase.Atoms
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor.get_atoms(self, **kwargs)