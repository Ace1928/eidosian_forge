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
class Screening(AbivarAble):
    """
    This object defines the parameters used for the
    computation of the screening function.
    """
    _WTYPES = dict(RPA=0)
    _SC_MODES = dict(one_shot=0, energy_only=1, wavefunctions=2)

    def __init__(self, ecuteps, nband, w_type='RPA', sc_mode='one_shot', hilbert=None, ecutwfn=None, inclvkb=2):
        """
        Args:
            ecuteps: Cutoff energy for the screening (Ha units).
            nband Number of bands for the Green's function
            w_type: Screening type
            sc_mode: Self-consistency mode.
            hilbert: Instance of HilbertTransform defining the parameters for the Hilber transform method.
            ecutwfn: Cutoff energy for the wavefunctions (Default: ecutwfn == ecut).
            inclvkb: Option for the treatment of the dipole matrix elements (NC pseudos).
        """
        if w_type not in self._WTYPES:
            raise ValueError(f'W_TYPE: {w_type} is not supported')
        if sc_mode not in self._SC_MODES:
            raise ValueError(f'Self-consistecy mode {sc_mode} is not supported')
        self.ecuteps = ecuteps
        self.nband = nband
        self.w_type = w_type
        self.sc_mode = sc_mode
        self.ecutwfn = ecutwfn
        self.inclvkb = inclvkb
        if hilbert is not None:
            raise NotImplementedError('Hilber transform not coded yet')
            self.hilbert = hilbert
        self.gwpara = 2
        self.awtr = 1
        self.symchi = 1
        self.optdriver = 3

    @property
    def use_hilbert(self):
        """True if we are using the Hilbert transform method."""
        return hasattr(self, 'hilbert')

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        abivars = {'ecuteps': self.ecuteps, 'ecutwfn': self.ecutwfn, 'inclvkb': self.inclvkb, 'gwpara': self.gwpara, 'awtr': self.awtr, 'symchi': self.symchi, 'nband': self.nband, 'optdriver': self.optdriver}
        if self.use_hilbert:
            abivars.update(self.hilbert.to_abivars())
        return abivars