from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
class RelaxationMethod(AbivarAble, MSONable):
    """
    This object stores the variables for the (constrained) structural optimization
    ionmov and optcell specify the type of relaxation.
    The other variables are optional and their use depend on ionmov and optcell.
    A None value indicates that we use abinit default. Default values can
    be modified by passing them to the constructor.
    The set of variables are constructed in to_abivars depending on ionmov and optcell.
    """
    _default_vars = dict(ionmov=MANDATORY, optcell=MANDATORY, ntime=80, dilatmx=1.05, ecutsm=0.5, strfact=None, tolmxf=None, strtarget=None, atoms_constraints={})
    IONMOV_DEFAULT = 3
    OPTCELL_DEFAULT = 2

    def __init__(self, *args, **kwargs):
        """Initialize object."""
        self.abivars = {**self._default_vars}
        self.abivars.update(*args, **kwargs)
        self.abivars = AttrDict(self.abivars)
        for key in self.abivars:
            if key not in self._default_vars:
                raise ValueError(f'{type(self).__name__}: No default value has been provided for key={key!r}')
        for key in self.abivars:
            if key is MANDATORY:
                raise ValueError(f'{type(self).__name__}: No default value has been provided for the mandatory key={key!r}')

    @classmethod
    def atoms_only(cls, atoms_constraints=None):
        """Relax atomic positions, keep unit cell fixed."""
        if atoms_constraints is None:
            return cls(ionmov=cls.IONMOV_DEFAULT, optcell=0)
        return cls(ionmov=cls.IONMOV_DEFAULT, optcell=0, atoms_constraints=atoms_constraints)

    @classmethod
    def atoms_and_cell(cls, atoms_constraints=None):
        """Relax atomic positions as well as unit cell."""
        if atoms_constraints is None:
            return cls(ionmov=cls.IONMOV_DEFAULT, optcell=cls.OPTCELL_DEFAULT)
        return cls(ionmov=cls.IONMOV_DEFAULT, optcell=cls.OPTCELL_DEFAULT, atoms_constraints=atoms_constraints)

    @property
    def move_atoms(self):
        """True if atoms must be moved."""
        return self.abivars.ionmov != 0

    @property
    def move_cell(self):
        """True if lattice parameters must be optimized."""
        return self.abivars.optcell != 0

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        out_vars = {'ionmov': self.abivars.ionmov, 'optcell': self.abivars.optcell, 'ntime': self.abivars.ntime}
        if self.move_atoms:
            out_vars.update({'tolmxf': self.abivars.tolmxf})
        if self.abivars.atoms_constraints:
            raise NotImplementedError('')
            out_vars.update(self.abivars.atoms_constraints.to_abivars())
        if self.move_cell:
            out_vars.update(dilatmx=self.abivars.dilatmx, ecutsm=self.abivars.ecutsm, strfact=self.abivars.strfact, strtarget=self.abivars.strtarget)
        return out_vars

    def as_dict(self):
        """Convert object to dict."""
        dct = dict(self._default_vars)
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build object from dictionary."""
        dct = dct.copy()
        dct.pop('@module', None)
        dct.pop('@class', None)
        return cls(**dct)