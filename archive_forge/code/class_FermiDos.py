from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
class FermiDos(Dos, MSONable):
    """This wrapper class helps relate the density of states, doping levels
    (i.e. carrier concentrations) and corresponding fermi levels. A negative
    doping concentration indicates the majority carriers are electrons
    (n-type doping); a positive doping concentration indicates holes are the
    majority carriers (p-type doping).
    """

    def __init__(self, dos: Dos, structure: Structure | None=None, nelecs: float | None=None, bandgap: float | None=None) -> None:
        """
        Args:
            dos: Pymatgen Dos object.
            structure: A structure. If not provided, the structure
                of the dos object will be used. If the dos does not have an
                associated structure object, an error will be thrown.
            nelecs: The number of electrons included in the energy range of
                dos. It is used for normalizing the densities. Default is the total
                number of electrons in the structure.
            bandgap: If set, the energy values are scissored so that the electronic
                band gap matches this value.
        """
        super().__init__(dos.efermi, energies=dos.energies, densities={k: np.array(d) for k, d in dos.densities.items()})
        if structure is None:
            if hasattr(dos, 'structure'):
                structure = dos.structure
            else:
                raise ValueError('Structure object is not provided and not present in dos')
        self.structure = structure
        self.nelecs = nelecs or self.structure.composition.total_electrons
        self.volume = self.structure.volume
        self.energies = np.array(dos.energies)
        self.de = np.hstack((self.energies[1:], self.energies[-1])) - self.energies
        tdos = np.array(self.get_densities())
        self.tdos = tdos * self.nelecs / (tdos * self.de)[self.energies <= self.efermi].sum()
        ecbm, evbm = self.get_cbm_vbm()
        self.idx_vbm = int(np.argmin(abs(self.energies - evbm)))
        self.idx_cbm = int(np.argmin(abs(self.energies - ecbm)))
        self.A_to_cm = 1e-08
        if bandgap:
            eref = self.efermi if evbm < self.efermi < ecbm else (evbm + ecbm) / 2.0
            idx_fermi = int(np.argmin(abs(self.energies - eref)))
            if idx_fermi == self.idx_vbm:
                idx_fermi += 1
            self.energies[:idx_fermi] -= (bandgap - (ecbm - evbm)) / 2.0
            self.energies[idx_fermi:] += (bandgap - (ecbm - evbm)) / 2.0

    def get_doping(self, fermi_level: float, temperature: float) -> float:
        """Calculate the doping (majority carrier concentration) at a given
        Fermi level  and temperature. A simple Left Riemann sum is used for
        integrating the density of states over energy & equilibrium Fermi-Dirac
        distribution.

        Args:
            fermi_level: The fermi_level level in eV.
            temperature: The temperature in Kelvin.

        Returns:
            The doping concentration in units of 1/cm^3. Negative values
            indicate that the majority carriers are electrons (n-type doping)
            whereas positive values indicates the majority carriers are holes
            (p-type doping).
        """
        cb_integral = np.sum(self.tdos[self.idx_cbm:] * f0(self.energies[self.idx_cbm:], fermi_level, temperature) * self.de[self.idx_cbm:], axis=0)
        vb_integral = np.sum(self.tdos[:self.idx_vbm + 1] * f0(-self.energies[:self.idx_vbm + 1], -fermi_level, temperature) * self.de[:self.idx_vbm + 1], axis=0)
        return (vb_integral - cb_integral) / (self.volume * self.A_to_cm ** 3)

    def get_fermi_interextrapolated(self, concentration: float, temperature: float, warn: bool=True, c_ref: float=10000000000.0, **kwargs) -> float:
        """Similar to get_fermi except that when get_fermi fails to converge,
        an interpolated or extrapolated fermi is returned with the assumption
        that the Fermi level changes linearly with log(abs(concentration)).

        Args:
            concentration: The doping concentration in 1/cm^3. Negative values
                represent n-type doping and positive values represent p-type
                doping.
            temperature: The temperature in Kelvin.
            warn: Whether to give a warning the first time the fermi cannot be
                found.
            c_ref: A doping concentration where get_fermi returns a
                value without error for both c_ref and -c_ref.
            **kwargs: Keyword arguments passed to the get_fermi function.

        Returns:
            The Fermi level. Note, the value is possibly interpolated or
            extrapolated and must be used with caution.
        """
        try:
            return self.get_fermi(concentration, temperature, **kwargs)
        except ValueError as exc:
            if warn:
                warnings.warn(str(exc))
            if abs(concentration) < c_ref:
                if abs(concentration) < 1e-10:
                    concentration = 1e-10
                f2 = self.get_fermi_interextrapolated(max(10, abs(concentration) * 10.0), temperature, warn=False, **kwargs)
                f1 = self.get_fermi_interextrapolated(-max(10, abs(concentration) * 10.0), temperature, warn=False, **kwargs)
                c2 = np.log(abs(1 + self.get_doping(f2, temperature)))
                c1 = -np.log(abs(1 + self.get_doping(f1, temperature)))
                slope = (f2 - f1) / (c2 - c1)
                return f2 + slope * (np.sign(concentration) * np.log(abs(1 + concentration)) - c2)
            f_ref = self.get_fermi_interextrapolated(np.sign(concentration) * c_ref, temperature, warn=False, **kwargs)
            f_new = self.get_fermi_interextrapolated(concentration / 10.0, temperature, warn=False, **kwargs)
            clog = np.sign(concentration) * np.log(abs(concentration))
            c_new_log = np.sign(concentration) * np.log(abs(self.get_doping(f_new, temperature)))
            slope = (f_new - f_ref) / (c_new_log - np.sign(concentration) * 10.0)
            return f_new + slope * (clog - c_new_log)

    def get_fermi(self, concentration: float, temperature: float, rtol: float=0.01, nstep: int=50, step: float=0.1, precision: int=8) -> float:
        """Finds the Fermi level at which the doping concentration at the given
        temperature (T) is equal to concentration. A greedy algorithm is used
        where the relative error is minimized by calculating the doping at a
        grid which continually becomes finer.

        Args:
            concentration: The doping concentration in 1/cm^3. Negative values
                represent n-type doping and positive values represent p-type
                doping.
            temperature: The temperature in Kelvin.
            rtol: The maximum acceptable relative error.
            nstep: The number of steps checked around a given Fermi level.
            step: Initial step in energy when searching for the Fermi level.
            precision: Essentially the decimal places of calculated Fermi level.

        Raises:
            ValueError: If the Fermi level cannot be found.

        Returns:
            The Fermi level in eV. Note that this is different from the default
            dos.efermi.
        """
        fermi = self.efermi
        relative_error = [float('inf')]
        for _ in range(precision):
            fermi_range = np.arange(-nstep, nstep + 1) * step + fermi
            calc_doping = np.array([self.get_doping(fermi_lvl, temperature) for fermi_lvl in fermi_range])
            relative_error = np.abs(calc_doping / concentration - 1.0)
            fermi = fermi_range[np.argmin(relative_error)]
            step /= 10.0
        if min(relative_error) > rtol:
            raise ValueError(f'Could not find fermi within {rtol:.1%} of concentration={concentration!r}')
        return fermi

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns Dos object from dict representation of Dos."""
        dos = Dos(dct['efermi'], dct['energies'], {Spin(int(k)): v for k, v in dct['densities'].items()})
        return cls(dos, structure=Structure.from_dict(dct['structure']), nelecs=dct['nelecs'])

    def as_dict(self) -> dict:
        """JSON-serializable dict representation of Dos."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'efermi': self.efermi, 'energies': self.energies.tolist(), 'densities': {str(spin): dens.tolist() for spin, dens in self.densities.items()}, 'structure': self.structure, 'nelecs': self.nelecs}