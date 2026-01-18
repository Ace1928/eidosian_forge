import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def _check_atom_properties(atoms, atoms_new, params):
    """ Checks that the properties of atoms is equal to the properties
    of atoms_new, and the parameters of atoms_new.calc is equal to params."""
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.get_masses(), atoms.get_masses())
    assert np.allclose(atoms_new.positions, atoms.positions, atol=0.001)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, atoms.cell)
    new_params = atoms_new.calc.parameters
    new_params_to_check = copy.deepcopy(new_params)
    params_to_check = copy.deepcopy(params)
    if 'basis_set' in params:
        params_to_check['basis_set'] = params_to_check['basis_set'].split('\n')
        params_to_check['basis_set'] = [line.strip() for line in params_to_check['basis_set']]
        new_params_to_check['basis_set'] = new_params_to_check['basis_set'].strip().split('\n')
    for key, value in new_params_to_check.items():
        params_equal = new_params_to_check.get(key) == params_to_check.get(key)
        if isinstance(params_equal, np.ndarray):
            assert (new_params_to_check.get(key) == params_to_check.get(key)).all()
        else:
            assert new_params_to_check.get(key) == params_to_check.get(key)