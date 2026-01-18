from random import randint
from typing import Dict, Tuple, Any
import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.calculators.calculator import (get_calculator_class, all_properties,
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols, atomic_masses
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import decode
class AtomsRow:

    def __init__(self, dct):
        if isinstance(dct, dict):
            dct = dct.copy()
            if 'calculator_parameters' in dct:
                while isinstance(dct['calculator_parameters'], str):
                    dct['calculator_parameters'] = decode(dct['calculator_parameters'])
        else:
            dct = atoms2dict(dct)
        assert 'numbers' in dct
        self._constraints = dct.pop('constraints', [])
        self._constrained_forces = None
        self._data = dct.pop('data', {})
        kvp = dct.pop('key_value_pairs', {})
        self._keys = list(kvp.keys())
        self.__dict__.update(kvp)
        self.__dict__.update(dct)
        if 'cell' not in dct:
            self.cell = np.zeros((3, 3))
        if 'pbc' not in dct:
            self.pbc = np.zeros(3, bool)

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return (key for key in self.__dict__ if key[0] != '_')

    def get(self, key, default=None):
        """Return value of key if present or default if not."""
        return getattr(self, key, default)

    @property
    def key_value_pairs(self):
        """Return dict of key-value pairs."""
        return dict(((key, self.get(key)) for key in self._keys))

    def count_atoms(self):
        """Count atoms.

        Return dict mapping chemical symbol strings to number of atoms.
        """
        count = {}
        for symbol in self.symbols:
            count[symbol] = count.get(symbol, 0) + 1
        return count

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return '<AtomsRow: formula={0}, keys={1}>'.format(self.formula, ','.join(self._keys))

    @property
    def constraints(self):
        """List of constraints."""
        if not isinstance(self._constraints, list):
            cs = decode(self._constraints)
            self._constraints = []
            for c in cs:
                name = c.pop('__name__', None)
                if name:
                    c = {'name': name, 'kwargs': c}
                if c['name'].startswith('ase'):
                    c['name'] = c['name'].rsplit('.', 1)[1]
                self._constraints.append(c)
        return [dict2constraint(d) for d in self._constraints]

    @property
    def data(self):
        """Data dict."""
        if isinstance(self._data, str):
            self._data = decode(self._data)
        elif isinstance(self._data, bytes):
            from ase.db.core import bytes_to_object
            self._data = bytes_to_object(self._data)
        return FancyDict(self._data)

    @property
    def natoms(self):
        """Number of atoms."""
        return len(self.numbers)

    @property
    def formula(self):
        """Chemical formula string."""
        return Formula('', _tree=[(self.symbols, 1)]).format('metal')

    @property
    def symbols(self):
        """List of chemical symbols."""
        return [chemical_symbols[Z] for Z in self.numbers]

    @property
    def fmax(self):
        """Maximum atomic force."""
        forces = self.constrained_forces
        return (forces ** 2).sum(1).max() ** 0.5

    @property
    def constrained_forces(self):
        """Forces after applying constraints."""
        if self._constrained_forces is not None:
            return self._constrained_forces
        forces = self.forces
        constraints = self.constraints
        if constraints:
            forces = forces.copy()
            atoms = self.toatoms()
            for constraint in constraints:
                constraint.adjust_forces(atoms, forces)
        self._constrained_forces = forces
        return forces

    @property
    def smax(self):
        """Maximum stress tensor component."""
        return (self.stress ** 2).max() ** 0.5

    @property
    def mass(self):
        """Total mass."""
        if 'masses' in self:
            return self.masses.sum()
        return atomic_masses[self.numbers].sum()

    @property
    def volume(self):
        """Volume of unit cell."""
        if self.cell is None:
            return None
        vol = abs(np.linalg.det(self.cell))
        if vol == 0.0:
            raise AttributeError
        return vol

    @property
    def charge(self):
        """Total charge."""
        charges = self.get('inital_charges')
        if charges is None:
            return 0.0
        return charges.sum()

    def toatoms(self, attach_calculator=False, add_additional_information=False):
        """Create Atoms object."""
        atoms = Atoms(self.numbers, self.positions, cell=self.cell, pbc=self.pbc, magmoms=self.get('initial_magmoms'), charges=self.get('initial_charges'), tags=self.get('tags'), masses=self.get('masses'), momenta=self.get('momenta'), constraint=self.constraints)
        if attach_calculator:
            params = self.get('calculator_parameters', {})
            atoms.calc = get_calculator_class(self.calculator)(**params)
        else:
            results = {}
            for prop in all_properties:
                if prop in self:
                    results[prop] = self[prop]
            if results:
                atoms.calc = SinglePointCalculator(atoms, **results)
                atoms.calc.name = self.get('calculator', 'unknown')
        if add_additional_information:
            atoms.info = {}
            atoms.info['unique_id'] = self.unique_id
            if self._keys:
                atoms.info['key_value_pairs'] = self.key_value_pairs
            data = self.get('data')
            if data:
                atoms.info['data'] = data
        return atoms