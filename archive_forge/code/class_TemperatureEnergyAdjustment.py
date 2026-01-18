from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
class TemperatureEnergyAdjustment(EnergyAdjustment):
    """An energy adjustment applied to a ComputedEntry based on the temperature.
    Used, for example, to add entropy to DFT energies.
    """

    def __init__(self, adj_per_deg, temp, n_atoms, uncertainty_per_deg=np.nan, name='', cls=None, description='Temperature-based energy adjustment'):
        """
        Args:
            adj_per_deg: float, energy adjustment to apply per degree K, in eV/atom
            temp: float, temperature in Kelvin
            n_atoms: float or int, number of atoms
            uncertainty_per_deg: float, uncertainty in energy adjustment to apply per degree K,
                in eV/atom. (Default: np.nan)
            name: str, human-readable name of the energy adjustment.
                (Default: "")
            cls: dict, Serialized Compatibility class used to generate the energy
                adjustment. (Default: None)
            description: str, human-readable explanation of the energy adjustment.
        """
        self._adj_per_deg = adj_per_deg
        self.uncertainty_per_deg = uncertainty_per_deg
        self.temp = temp
        self.n_atoms = n_atoms
        self.name = name
        self.cls = cls or {}
        self.description = description

    @property
    def value(self):
        """Return the value of the energy correction in eV."""
        return self._adj_per_deg * self.temp * self.n_atoms

    @property
    def uncertainty(self):
        """Return the value of the energy adjustment in eV."""
        return self.uncertainty_per_deg * self.temp * self.n_atoms

    @property
    def explain(self):
        """Return an explanation of how the energy adjustment is calculated."""
        return f'{self.description} ({self._adj_per_deg:.4f} eV/K/atom x {self.temp} K x {self.n_atoms} atoms)'

    def normalize(self, factor: float) -> None:
        """Normalize energy adjustment (in place), dividing value/uncertainty by a
        factor.

        Args:
            factor: factor to divide by.
        """
        self.n_atoms /= factor