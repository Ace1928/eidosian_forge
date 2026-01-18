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
class ExcHamiltonian(AbivarAble):
    """This object contains the parameters for the solution of the Bethe-Salpeter equation."""
    _EXC_TYPES = dict(TDA=0, coupling=1)
    _ALGO2VAR = dict(direct_diago=1, haydock=2, cg=3)
    _COULOMB_MODES = ('diago', 'full', 'model_df')

    def __init__(self, bs_loband, nband, mbpt_sciss, coulomb_mode, ecuteps, spin_mode='polarized', mdf_epsinf=None, exc_type='TDA', algo='haydock', with_lf=True, bs_freq_mesh=None, zcut=None, **kwargs):
        """
        Args:
            bs_loband: Lowest band index (Fortran convention) used in the e-h  basis set.
                Can be scalar or array of shape (nsppol,). Must be >= 1 and <= nband
            nband: Max band index used in the e-h  basis set.
            mbpt_sciss: Scissors energy in Hartree.
            coulomb_mode: Treatment of the Coulomb term.
            ecuteps: Cutoff energy for W in Hartree.
            mdf_epsinf: Macroscopic dielectric function :math:`\\\\epsilon_\\\\inf` used in
                the model dielectric function.
            exc_type: Approximation used for the BSE Hamiltonian
            with_lf: True if local field effects are included <==> exchange term is included
            bs_freq_mesh: Frequency mesh for the macroscopic dielectric function (start, stop, step) in Ha.
            zcut: Broadening parameter in Ha.
            **kwargs:
                Extra keywords.
        """
        spin_mode = SpinMode.as_spinmode(spin_mode)
        try:
            bs_loband = np.reshape(bs_loband, spin_mode.nsppol)
        except ValueError:
            bs_loband = np.array(spin_mode.nsppol * [int(bs_loband)])
        self.bs_loband = bs_loband
        self.nband = nband
        self.mbpt_sciss = mbpt_sciss
        self.coulomb_mode = coulomb_mode
        assert coulomb_mode in self._COULOMB_MODES
        self.ecuteps = ecuteps
        self.mdf_epsinf = mdf_epsinf
        self.exc_type = exc_type
        assert exc_type in self._EXC_TYPES
        self.algo = algo
        assert algo in self._ALGO2VAR
        self.with_lf = with_lf
        self.bs_freq_mesh = np.array(bs_freq_mesh) if bs_freq_mesh is not None else bs_freq_mesh
        self.zcut = zcut
        self.optdriver = 99
        self.kwargs = kwargs
        if any(bs_loband < 0):
            raise ValueError(f'bs_loband <= 0 while it is {bs_loband}')
        if any(bs_loband >= nband):
            raise ValueError(f'(bs_loband={bs_loband!r}) >= (nband={nband!r})')

    @property
    def inclvkb(self):
        """Treatment of the dipole matrix element (NC pseudos, default is 2)."""
        return self.kwargs.get('inclvkb', 2)

    @property
    def use_haydock(self):
        """True if we are using the Haydock iterative technique."""
        return self.algo == 'haydock'

    @property
    def use_cg(self):
        """True if we are using the conjugate gradient method."""
        return self.algo == 'cg'

    @property
    def use_direct_diago(self):
        """True if we are performing the direct diagonalization of the BSE Hamiltonian."""
        return self.algo == 'direct_diago'

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        abivars = {'bs_calctype': 1, 'bs_loband': self.bs_loband, 'mbpt_sciss': self.mbpt_sciss, 'ecuteps': self.ecuteps, 'bs_algorithm': self._ALGO2VAR[self.algo], 'bs_coulomb_term': 21, 'mdf_epsinf': self.mdf_epsinf, 'bs_exchange_term': 1 if self.with_lf else 0, 'inclvkb': self.inclvkb, 'zcut': self.zcut, 'bs_freq_mesh': self.bs_freq_mesh, 'bs_coupling': self._EXC_TYPES[self.exc_type], 'optdriver': self.optdriver}
        if self.use_haydock:
            abivars.update(bs_haydock_niter=100, bs_hayd_term=0, bs_haydock_tol=[0.05, 0])
        elif self.use_direct_diago or self.use_cg:
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown algorithm for EXC: {self.algo}')
        abivars.update(self.kwargs)
        return abivars