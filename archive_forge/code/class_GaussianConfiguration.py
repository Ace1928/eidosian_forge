import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
class GaussianConfiguration:

    def __init__(self, atoms, parameters):
        self.atoms = atoms.copy()
        self.parameters = deepcopy(parameters)

    def get_atoms(self):
        return self.atoms

    def get_parameters(self):
        return self.parameters

    def get_calculator(self):
        calc = Gaussian(atoms=self.atoms, command=None, restart=None, ignore_bad_restart_file=Calculator._deprecated, label='Gaussian', directory='.', **self.parameters)
        return calc

    @staticmethod
    def parse_gaussian_input(fd):
        """Reads a gaussian input file into an atoms object and
        parameters dictionary.

        Parameters
        ----------
        fd: file-like
            Contains the contents of a  gaussian input file

        Returns
        ---------
        GaussianConfiguration
            Contains an atoms object created using the structural
            information from the input file.
            Contains a parameters dictionary, which stores any
            keywords and options found in the link-0 and route
            sections of the input file.
        """
        parameters = {}
        file_sections = _get_gaussian_in_sections(fd)
        parameters.update(_get_all_link0_params(file_sections['link0']))
        parameters.update(_get_all_route_params(file_sections['route']))
        parameters.update(_get_charge_mult(file_sections['charge_mult']))
        atoms, nuclear_props = _get_atoms_from_molspec(file_sections['mol_spec'])
        parameters.update(nuclear_props)
        parameters.update(_get_extra_section_params(file_sections['extra'], parameters, atoms))
        _validate_params(parameters)
        return GaussianConfiguration(atoms, parameters)