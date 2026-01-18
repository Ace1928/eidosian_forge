from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from pymatgen.core.units import kb as kb_ev
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1002/chem.201200497'), description='Supramolecular Binding Thermodynamics by Dispersion-Corrected Density Functional Theory')
def _get_quasirrho_thermo(self, mol: Molecule, mult: int, sigma_r: int, frequencies: list[float], elec_energy: float) -> None:
    """
        Calculate Quasi-RRHO thermochemistry

        Args:
            mol (Molecule): Pymatgen molecule
            mult (int): Spin multiplicity
            sigma_r (int): Rotational symmetry number
            frequencies (list): List of frequencies [cm^-1]
            elec_energy (float): Electronic energy [Ha]
        """
    mass: float = 0
    for site in mol:
        mass += site.specie.atomic_mass
    mass *= amu_to_kg
    vib_temps = [freq * light_speed * h_plank / kb for freq in frequencies if freq > 0]
    qt = (2 * np.pi * mass * kb * self.temp / (h_plank * h_plank)) ** (3 / 2) * kb * self.temp / self.press
    st = ideal_gas_const * (np.log(qt) + 5 / 2)
    et = 3 * ideal_gas_const * self.temp / 2
    se = ideal_gas_const * np.log(mult)
    Bav, i_eigen = get_avg_mom_inertia(mol)
    coords = mol.cart_coords
    v0 = coords[1] - coords[0]
    linear = True
    for coord in coords[1:]:
        theta = abs(np.dot(coord - coords[0], v0) / np.linalg.norm(coord - coords[0]) / np.linalg.norm(v0))
        if not isclose(theta, 1, abs_tol=0.0001):
            linear = False
    if linear:
        i = np.amax(i_eigen)
        qr = 8 * np.pi ** 2 * i * kb * self.temp / (sigma_r * (h_plank * h_plank))
        sr = ideal_gas_const * (np.log(qr) + 1)
        er = ideal_gas_const * self.temp
    else:
        rot_temps = [h_plank ** 2 / (np.pi ** 2 * kb * 8 * i) for i in i_eigen]
        qr = np.sqrt(np.pi) / sigma_r * self.temp ** (3 / 2) / np.sqrt(rot_temps[0] * rot_temps[1] * rot_temps[2])
        sr = ideal_gas_const * (np.log(qr) + 3 / 2)
        er = 3 * ideal_gas_const * self.temp / 2
    ev = 0
    sv_quasiRRHO = 0
    sv = 0
    for vt in vib_temps:
        ev += vt * (1 / 2 + 1 / (np.exp(vt / self.temp) - 1))
        sv_temp = vt / (self.temp * (np.exp(vt / self.temp) - 1)) - np.log(1 - np.exp(-vt / self.temp))
        sv += sv_temp
        mu = h_plank / (8 * np.pi ** 2 * vt * light_speed)
        mu_prime = mu * Bav / (mu + Bav)
        s_rotor = 1 / 2 + np.log(np.sqrt(8 * np.pi ** 3 * mu_prime * kb * self.temp / h_plank ** 2))
        weight = 1 / (1 + (self.v0 / vt) ** 4)
        sv_quasiRRHO += weight * sv_temp + (1 - weight) * s_rotor
    sv_quasiRRHO *= ideal_gas_const
    sv *= ideal_gas_const
    ev *= ideal_gas_const
    e_tot = (et + er + ev) * kcal_to_hartree / 1000
    self.h_corrected = e_tot + ideal_gas_const * self.temp * kcal_to_hartree / 1000
    self.entropy_ho = st + sr + sv + se
    self.free_energy_ho = elec_energy + self.h_corrected - self.temp * self.entropy_ho * kcal_to_hartree / 1000
    self.entropy_quasiRRHO = st + sr + sv_quasiRRHO + se
    self.free_energy_quasiRRHO = elec_energy + self.h_corrected - self.temp * self.entropy_quasiRRHO * kcal_to_hartree / 1000