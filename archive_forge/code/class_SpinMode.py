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
class SpinMode(namedtuple('SpinMode', 'mode nsppol nspinor nspden'), AbivarAble, MSONable):
    """
    Different configurations of the electron density as implemented in abinit:
    One can use as_spinmode to construct the object via SpinMode.as_spinmode
    (string) where string can assume the values:

        - polarized
        - unpolarized
        - afm (anti-ferromagnetic)
        - spinor (non-collinear magnetism)
        - spinor_nomag (non-collinear, no magnetism)
    """
    __slots__ = ()

    @classmethod
    def as_spinmode(cls, obj):
        """Converts obj into a `SpinMode` instance."""
        if isinstance(obj, cls):
            return obj
        try:
            return _mode_to_spin_vars[obj]
        except KeyError:
            raise KeyError(f'Wrong value for spin_mode: {obj}')

    def to_abivars(self):
        """Dictionary with Abinit input variables."""
        return {'nsppol': self.nsppol, 'nspinor': self.nspinor, 'nspden': self.nspden}

    def as_dict(self):
        """Convert object to dict."""
        out = {k: getattr(self, k) for k in self._fields}
        out.update({'@module': type(self).__module__, '@class': type(self).__name__})
        return out

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build object from dict."""
        return cls(**{key: dct[key] for key in dct if key in cls._fields})