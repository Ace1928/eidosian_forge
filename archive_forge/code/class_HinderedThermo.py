import os
import sys
import numpy as np
from ase import units
class HinderedThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the hindered
    translator and hindered rotor model where all but three degrees of
    freedom are treated as harmonic vibrations, two are treated as
    hindered translations, and one is treated as a hindered rotation.

    Inputs:

    vib_energies : list
        a list of all the vibrational energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of energies
        should match the number of degrees of freedom of the adsorbate;
        i.e., 3*n, where n is the number of atoms. Note that this class does
        not check that the user has supplied the correct number of energies.
        Units of energies are eV.
    trans_barrier_energy : float
        the translational energy barrier in eV. This is the barrier for an
        adsorbate to diffuse on the surface.
    rot_barrier_energy : float
        the rotational energy barrier in eV. This is the barrier for an
        adsorbate to rotate about an axis perpendicular to the surface.
    sitedensity : float
        density of surface sites in cm^-2
    rotationalminima : integer
        the number of equivalent minima for an adsorbate's full rotation.
        For example, 6 for an adsorbate on an fcc(111) top site
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this class
        can be interpreted as the energy corrections)
    mass : float
        the mass of the adsorbate in amu (if mass is unspecified, then it will
        be calculated from the atoms class)
    inertia : float
        the reduced moment of inertia of the adsorbate in amu*Ang^-2
        (if inertia is unspecified, then it will be calculated from the
        atoms class)
    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the adsorbate. This is the number of symmetric arms
        of the adsorbate and depends upon how it is bound to the surface.
        For example, propane bound through its end carbon has a symmetry
        number of 1 but propane bound through its middle carbon has a symmetry
        number of 2. (if symmetrynumber is unspecified, then the default is 1)
    """

    def __init__(self, vib_energies, trans_barrier_energy, rot_barrier_energy, sitedensity, rotationalminima, potentialenergy=0.0, mass=None, inertia=None, atoms=None, symmetrynumber=1):
        self.vib_energies = sorted(vib_energies, reverse=True)[:-3]
        self.trans_barrier_energy = trans_barrier_energy * units._e
        self.rot_barrier_energy = rot_barrier_energy * units._e
        self.area = 1.0 / sitedensity / 100.0 ** 2
        self.rotationalminima = rotationalminima
        self.potentialenergy = potentialenergy
        self.atoms = atoms
        self.symmetry = symmetrynumber
        if (mass or atoms) and (inertia or atoms):
            if mass:
                self.mass = mass * units._amu
            elif atoms:
                self.mass = np.sum(atoms.get_masses()) * units._amu
            if inertia:
                self.inertia = inertia * units._amu / units.m ** 2
            elif atoms:
                self.inertia = atoms.get_moments_of_inertia()[2] * units._amu / units.m ** 2
        else:
            raise RuntimeError('Either mass and inertia of the adsorbate must be specified or atoms must be specified.')
        if sum(np.iscomplex(self.vib_energies)):
            raise ValueError('Imaginary frequencies are present.')
        else:
            self.vib_energies = np.real(self.vib_energies)
        self.freq_t = np.sqrt(self.trans_barrier_energy / (2 * self.mass * self.area))
        self.freq_r = 1.0 / (2 * np.pi) * np.sqrt(self.rotationalminima ** 2 * self.rot_barrier_energy / (2 * self.inertia))

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy (including the zero point energy),
        in eV, in the hindered translator and hindered rotor model at a
        specified temperature (K)."""
        from scipy.special import iv
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        write('Internal energy components at T = %.2f K:' % temperature)
        write('=' * 31)
        U = 0.0
        write(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy
        T_t = units._k * temperature / (units._hplanck * self.freq_t)
        R_t = self.trans_barrier_energy / (units._hplanck * self.freq_t)
        dU_t = 2 * (-1.0 / 2 - 1.0 / T_t / (2 + 16 * R_t) + R_t / 2 / T_t - R_t / 2 / T_t * iv(1, R_t / 2 / T_t) / iv(0, R_t / 2 / T_t) + 1.0 / T_t / (np.exp(1.0 / T_t) - 1))
        dU_t *= units.kB * temperature
        write(fmt % ('E_trans', dU_t))
        U += dU_t
        T_r = units._k * temperature / (units._hplanck * self.freq_r)
        R_r = self.rot_barrier_energy / (units._hplanck * self.freq_r)
        dU_r = -1.0 / 2 - 1.0 / T_r / (2 + 16 * R_r) + R_r / 2 / T_r - R_r / 2 / T_r * iv(1, R_r / 2 / T_r) / iv(0, R_r / 2 / T_r) + 1.0 / T_r / (np.exp(1.0 / T_r) - 1)
        dU_r *= units.kB * temperature
        write(fmt % ('E_rot', dU_r))
        U += dU_r
        dU_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('E_vib', dU_v))
        U += dU_v
        dU_zpe = self.get_zero_point_energy()
        write(fmt % ('E_ZPE', dU_zpe))
        U += dU_zpe
        write('-' * 31)
        write(fmt % ('U', U))
        write('=' * 31)
        return U

    def get_zero_point_energy(self, verbose=True):
        """Returns the zero point energy, in eV, in the hindered
        translator and hindered rotor model"""
        zpe_t = 2 * (1.0 / 2 * self.freq_t * units._hplanck / units._e)
        zpe_r = 1.0 / 2 * self.freq_r * units._hplanck / units._e
        zpe_v = self.get_ZPE_correction()
        zpe = zpe_t + zpe_r + zpe_v
        return zpe

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, in the hindered translator
        and hindered rotor model at a specified temperature (K)."""
        from scipy.special import iv
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))
        S = 0.0
        T_t = units._k * temperature / (units._hplanck * self.freq_t)
        R_t = self.trans_barrier_energy / (units._hplanck * self.freq_t)
        S_t = 2 * (-1.0 / 2 + 1.0 / 2 * np.log(np.pi * R_t / T_t) - R_t / 2 / T_t * iv(1, R_t / 2 / T_t) / iv(0, R_t / 2 / T_t) + np.log(iv(0, R_t / 2 / T_t)) + 1.0 / T_t / (np.exp(1.0 / T_t) - 1) - np.log(1 - np.exp(-1.0 / T_t)))
        S_t *= units.kB
        write(fmt % ('S_trans', S_t, S_t * temperature))
        S += S_t
        T_r = units._k * temperature / (units._hplanck * self.freq_r)
        R_r = self.rot_barrier_energy / (units._hplanck * self.freq_r)
        S_r = -1.0 / 2 + 1.0 / 2 * np.log(np.pi * R_r / T_r) - np.log(self.symmetry) - R_r / 2 / T_r * iv(1, R_r / 2 / T_r) / iv(0, R_r / 2 / T_r) + np.log(iv(0, R_r / 2 / T_r)) + 1.0 / T_r / (np.exp(1.0 / T_r) - 1) - np.log(1 - np.exp(-1.0 / T_r))
        S_r *= units.kB
        write(fmt % ('S_rot', S_r, S_r * temperature))
        S += S_r
        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_vib', S_v, S_v * temperature))
        S += S_v
        N_over_A = np.exp(1.0 / 3) * (10.0 ** 5 / (units._k * temperature)) ** (2.0 / 3)
        S_c = 1 - np.log(N_over_A) - np.log(self.area)
        S_c *= units.kB
        write(fmt % ('S_con', S_c, S_c * temperature))
        S += S_c
        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, in the hindered
        translator and hindered rotor model at a specified temperature
        (K)."""
        self.verbose = True
        write = self._vprint
        U = self.get_internal_energy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S
        write('')
        write('Free energy components at T = %.2f K:' % temperature)
        write('=' * 23)
        fmt = '%5s%15.3f eV'
        write(fmt % ('U', U))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('F', F))
        write('=' * 23)
        return F