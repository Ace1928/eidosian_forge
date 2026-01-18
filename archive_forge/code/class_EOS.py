from __future__ import annotations
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import leastsq, minimize
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
class EOS:
    """
    Convenient wrapper. Retained in its original state to ensure backward
    compatibility.

    Fit equation of state for bulk systems.

    The following equations are supported:

        murnaghan: PRB 28, 5480 (1983)

        birch: Intermetallic compounds: Principles and Practice, Vol I:
            Principles. pages 195-210

        birch_murnaghan: PRB 70, 224107

        pourier_tarantola: PRB 70, 224107

        vinet: PRB 70, 224107

        deltafactor

        numerical_eos: 10.1103/PhysRevB.90.174107.

    Usage:

       eos = EOS(eos_name='murnaghan')
       eos_fit = eos.fit(volumes, energies)
       eos_fit.plot()
    """
    MODELS = dict(murnaghan=Murnaghan, birch=Birch, birch_murnaghan=BirchMurnaghan, pourier_tarantola=PourierTarantola, vinet=Vinet, deltafactor=DeltaFactor, numerical_eos=NumericalEOS)

    def __init__(self, eos_name='murnaghan'):
        """
        Args:
            eos_name (str): Type of EOS to fit.
        """
        if eos_name not in self.MODELS:
            raise EOSError(f'The equation of state {eos_name!r} is not supported. Please choose one from the following list: {list(self.MODELS)}')
        self._eos_name = eos_name
        self.model = self.MODELS[eos_name]

    def fit(self, volumes, energies):
        """
        Fit energies as function of volumes.

        Args:
            volumes (list/np.array)
            energies (list/np.array)

        Returns:
            EOSBase: EOSBase object
        """
        eos_fit = self.model(np.array(volumes), np.array(energies))
        eos_fit.fit()
        return eos_fit