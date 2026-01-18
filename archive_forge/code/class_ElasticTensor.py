from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
class ElasticTensor(NthOrderElasticTensor):
    """
    This class extends Tensor to describe the 3x3x3x3 second-order elastic tensor,
    C_{ijkl}, with various methods for estimating other properties derived from the second
    order elastic tensor (e. g. bulk modulus, shear modulus, Young's modulus, Poisson's ratio)
    in units of eV/A^3.
    """

    def __new__(cls, input_array, tol: float=0.0001) -> Self:
        """
        Create an ElasticTensor object. The constructor throws an error if the shape of
        the input_matrix argument is not 3x3x3x3, i. e. in true tensor notation. Issues a
        warning if the input_matrix argument does not satisfy standard symmetries. Note
        that the constructor uses __new__ rather than __init__ according to the standard
        method of subclassing numpy ndarrays.

        Args:
            input_array (3x3x3x3 array-like): the 3x3x3x3 array-like representing the elastic tensor

            tol (float): tolerance for initial symmetry test of tensor
        """
        obj = super().__new__(cls, input_array, check_rank=4, tol=tol)
        return obj.view(cls)

    @property
    def compliance_tensor(self):
        """
        Returns the Voigt notation compliance tensor, which is the matrix
        inverse of the Voigt notation elastic tensor.
        """
        s_voigt = np.linalg.inv(self.voigt)
        return ComplianceTensor.from_voigt(s_voigt)

    @property
    def k_voigt(self) -> float:
        """Returns the K_v bulk modulus (in eV/A^3)."""
        return self.voigt[:3, :3].mean()

    @property
    def g_voigt(self) -> float:
        """Returns the G_v shear modulus (in eV/A^3)."""
        return (2 * self.voigt[:3, :3].trace() - np.triu(self.voigt[:3, :3]).sum() + 3 * self.voigt[3:, 3:].trace()) / 15.0

    @property
    def k_reuss(self) -> float:
        """Returns the K_r bulk modulus (in eV/A^3)."""
        return 1 / self.compliance_tensor.voigt[:3, :3].sum()

    @property
    def g_reuss(self) -> float:
        """Returns the G_r shear modulus (in eV/A^3)."""
        return 15 / (8 * self.compliance_tensor.voigt[:3, :3].trace() - 4 * np.triu(self.compliance_tensor.voigt[:3, :3]).sum() + 3 * self.compliance_tensor.voigt[3:, 3:].trace())

    @property
    def k_vrh(self) -> float:
        """Returns the K_vrh (Voigt-Reuss-Hill) average bulk modulus (in eV/A^3)."""
        return 0.5 * (self.k_voigt + self.k_reuss)

    @property
    def g_vrh(self) -> float:
        """Returns the G_vrh (Voigt-Reuss-Hill) average shear modulus (in eV/A^3)."""
        return 0.5 * (self.g_voigt + self.g_reuss)

    @property
    def y_mod(self) -> float:
        """
        Calculates Young's modulus (in SI units) using the
        Voigt-Reuss-Hill averages of bulk and shear moduli.
        """
        return 9000000000.0 * self.k_vrh * self.g_vrh / (3 * self.k_vrh + self.g_vrh)

    def directional_poisson_ratio(self, n: ArrayLike, m: ArrayLike, tol: float=1e-08) -> float:
        """
        Calculates the poisson ratio for a specific direction
        relative to a second, orthogonal direction.

        Args:
            n (3-d vector): principal direction
            m (3-d vector): secondary direction orthogonal to n
            tol (float): tolerance for testing of orthogonality
        """
        n, m = (get_uvec(n), get_uvec(m))
        if not np.abs(np.dot(n, m)) < tol:
            raise ValueError('n and m must be orthogonal')
        v = self.compliance_tensor.einsum_sequence([n] * 2 + [m] * 2)
        v *= -1 / self.compliance_tensor.einsum_sequence([n] * 4)
        return v

    def directional_elastic_mod(self, n) -> float:
        """Calculates directional elastic modulus for a specific vector."""
        n = get_uvec(n)
        return self.einsum_sequence([n] * 4)

    @raise_if_unphysical
    def trans_v(self, structure: Structure) -> float:
        """
        Calculates transverse sound velocity using the
        Voigt-Reuss-Hill average bulk modulus.

        Args:
            structure: pymatgen structure object

        Returns:
            float: transverse sound velocity (in SI units)
        """
        n_sites = len(structure)
        n_atoms = structure.composition.num_atoms
        weight = float(structure.composition.weight)
        mass_density = 1660.5 * n_sites * weight / (n_atoms * structure.volume)
        if self.g_vrh < 0:
            raise ValueError('k_vrh or g_vrh is negative, sound velocity is undefined')
        return (1000000000.0 * self.g_vrh / mass_density) ** 0.5

    @raise_if_unphysical
    def long_v(self, structure: Structure) -> float:
        """
        Calculates longitudinal sound velocity using the Voigt-Reuss-Hill average bulk modulus.

        Args:
            structure: pymatgen structure object

        Returns:
            float: longitudinal sound velocity (in SI units)
        """
        n_sites = len(structure)
        n_atoms = structure.composition.num_atoms
        weight = float(structure.composition.weight)
        mass_density = 1660.5 * n_sites * weight / (n_atoms * structure.volume)
        if self.g_vrh < 0:
            raise ValueError('k_vrh or g_vrh is negative, sound velocity is undefined')
        return (1000000000.0 * (self.k_vrh + 4 / 3 * self.g_vrh) / mass_density) ** 0.5

    @raise_if_unphysical
    def snyder_ac(self, structure: Structure) -> float:
        """Calculates Snyder's acoustic sound velocity.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Snyder's acoustic sound velocity (in SI units)
        """
        n_sites = len(structure)
        n_atoms = structure.composition.num_atoms
        site_density = 1e+30 * n_sites / structure.volume
        tot_mass = sum((spec.atomic_mass for spec in structure.species))
        avg_mass = 1.6605e-27 * tot_mass / n_atoms
        return 0.38483 * avg_mass * ((self.long_v(structure) + 2 * self.trans_v(structure)) / 3) ** 3.0 / (300 * site_density ** (-2 / 3) * n_sites ** (1 / 3))

    @raise_if_unphysical
    def snyder_opt(self, structure: Structure) -> float:
        """
        Calculates Snyder's optical sound velocity (in SI units).

        Args:
            structure: pymatgen Structure object

        Returns:
            float: Snyder's optical sound velocity (in SI units)
        """
        n_sites = len(structure)
        site_density = 1e+30 * n_sites / structure.volume
        return 1.66914e-23 * (self.long_v(structure) + 2 * self.trans_v(structure)) / 3.0 / site_density ** (-2 / 3) * (1 - n_sites ** (-1 / 3))

    @raise_if_unphysical
    def snyder_total(self, structure: Structure) -> float:
        """
        Calculates Snyder's total sound velocity (in SI units).

        Args:
            structure: pymatgen structure object

        Returns:
            float: Snyder's total sound velocity (in SI units)
        """
        return self.snyder_ac(structure) + self.snyder_opt(structure)

    @raise_if_unphysical
    def clarke_thermalcond(self, structure: Structure) -> float:
        """Calculates Clarke's thermal conductivity.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Clarke's thermal conductivity (in SI units)
        """
        n_sites = len(structure)
        tot_mass = sum((spec.atomic_mass for spec in structure.species))
        n_atoms = structure.composition.num_atoms
        weight = float(structure.composition.weight)
        avg_mass = 1.6605e-27 * tot_mass / n_atoms
        mass_density = 1660.5 * n_sites * weight / (n_atoms * structure.volume)
        return 0.87 * 1.3806e-23 * avg_mass ** (-2 / 3) * mass_density ** (1 / 6) * self.y_mod ** 0.5

    @raise_if_unphysical
    def cahill_thermalcond(self, structure: Structure) -> float:
        """Calculate Cahill's thermal conductivity.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Cahill's thermal conductivity (in SI units)
        """
        n_sites = len(structure)
        site_density = 1e+30 * n_sites / structure.volume
        return 1.3806e-23 / 2.48 * site_density ** (2 / 3) * (self.long_v(structure) + 2 * self.trans_v(structure))

    @due.dcite(Doi('10.1039/C7EE03256K'), description='Minimum thermal conductivity in the context of diffusion-mediated thermal transport')
    @raise_if_unphysical
    def agne_diffusive_thermalcond(self, structure: Structure) -> float:
        """Calculates Agne's diffusive thermal conductivity.

        Please cite the original authors if using this method
        M. T. Agne, R. Hanus, G. J. Snyder, Energy Environ. Sci. 2018, 11, 609-616.
        DOI: https://doi.org/10.1039/C7EE03256K

        Args:
            structure: pymatgen structure object

        Returns:
            float: Agne's diffusive thermal conductivity (in SI units)
        """
        n_sites = len(structure)
        site_density = 1e+30 * n_sites / structure.volume
        return 0.76 * site_density ** (2 / 3) * 1.3806e-23 * (1 / 3 * (2 * self.trans_v(structure) + self.long_v(structure)))

    @raise_if_unphysical
    def debye_temperature(self, structure: Structure) -> float:
        """
        Estimates the Debye temperature from longitudinal and transverse sound velocities.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Debye temperature (in SI units)
        """
        v0 = structure.volume * 1e-30 / len(structure)
        vl, vt = (self.long_v(structure), self.trans_v(structure))
        vm = 3 ** (1 / 3) * (1 / vl ** 3 + 2 / vt ** 3) ** (-1 / 3)
        return 1.05457e-34 / 1.38065e-23 * vm * (6 * np.pi ** 2 / v0) ** (1 / 3)

    @property
    def universal_anisotropy(self) -> float:
        """Returns the universal anisotropy value."""
        return 5 * self.g_voigt / self.g_reuss + self.k_voigt / self.k_reuss - 6.0

    @property
    def homogeneous_poisson(self) -> float:
        """Returns the homogeneous poisson ratio."""
        return (1 - 2 / 3 * self.g_vrh / self.k_vrh) / (2 + 2 / 3 * self.g_vrh / self.k_vrh)

    def green_kristoffel(self, u) -> float:
        """Returns the Green-Kristoffel tensor for a second-order tensor."""
        return self.einsum_sequence([u, u], 'ijkl,i,l')

    @property
    def property_dict(self):
        """Returns a dictionary of properties derived from the elastic tensor."""
        props = ('k_voigt', 'k_reuss', 'k_vrh', 'g_voigt', 'g_reuss', 'g_vrh', 'universal_anisotropy', 'homogeneous_poisson', 'y_mod')
        return {prop: getattr(self, prop) for prop in props}

    def get_structure_property_dict(self, structure: Structure, include_base_props: bool=True, ignore_errors: bool=False) -> dict[str, float | Structure | None]:
        """
        Returns a dictionary of properties derived from the elastic tensor
        and an associated structure.

        Args:
            structure (Structure): structure object for which to calculate
                associated properties
            include_base_props (bool): whether to include base properties,
                like k_vrh, etc.
            ignore_errors (bool): if set to true, will set problem properties
                that depend on a physical tensor to None, defaults to False
        """
        s_props = ('trans_v', 'long_v', 'snyder_ac', 'snyder_opt', 'snyder_total', 'clarke_thermalcond', 'cahill_thermalcond', 'debye_temperature')
        sp_dict: dict[str, float | Structure | None]
        if ignore_errors and (self.k_vrh < 0 or self.g_vrh < 0):
            sp_dict = dict.fromkeys(s_props)
        else:
            sp_dict = {prop: getattr(self, prop)(structure) for prop in s_props}
        sp_dict['structure'] = structure
        if include_base_props:
            sp_dict.update(self.property_dict)
        return sp_dict

    @classmethod
    def from_pseudoinverse(cls, strains, stresses) -> Self:
        """
        Class method to fit an elastic tensor from stress/strain
        data. Method uses Moore-Penrose pseudo-inverse to invert
        the s = C*e equation with elastic tensor, stress, and
        strain in voigt notation.

        Args:
            stresses (Nx3x3 array-like): list or array of stresses
            strains (Nx3x3 array-like): list or array of strains
        """
        warnings.warn('Pseudo-inverse fitting of Strain/Stress lists may yield questionable results from vasp data, use with caution.')
        stresses = np.array([Stress(stress).voigt for stress in stresses])
        with warnings.catch_warnings():
            strains = np.array([Strain(strain).voigt for strain in strains])
        voigt_fit = np.transpose(np.dot(np.linalg.pinv(strains), stresses))
        return cls.from_voigt(voigt_fit)

    @classmethod
    def from_independent_strains(cls, strains, stresses, eq_stress=None, vasp=False, tol: float=1e-10) -> Self:
        """
        Constructs the elastic tensor least-squares fit of independent strains

        Args:
            strains (list of Strains): list of strain objects to fit
            stresses (list of Stresses): list of stress objects to use in fit
                corresponding to the list of strains
            eq_stress (Stress): equilibrium stress to use in fitting
            vasp (bool): flag for whether the stress tensor should be
                converted based on vasp units/convention for stress
            tol (float): tolerance for removing near-zero elements of the
                resulting tensor.
        """
        strain_states = [tuple(ss) for ss in np.eye(6)]
        ss_dict = get_strain_state_dict(strains, stresses, eq_stress=eq_stress)
        if not set(strain_states) <= set(ss_dict):
            raise ValueError(f'Missing independent strain states: {set(strain_states) - set(ss_dict)}')
        if len(set(ss_dict) - set(strain_states)) > 0:
            warnings.warn('Extra strain states in strain-stress pairs are neglected in independent strain fitting')
        c_ij = np.zeros((6, 6))
        for ii in range(6):
            strains = ss_dict[strain_states[ii]]['strains']
            stresses = ss_dict[strain_states[ii]]['stresses']
            for jj in range(6):
                c_ij[ii, jj] = np.polyfit(strains[:, ii], stresses[:, jj], 1)[0]
        if vasp:
            c_ij *= -0.1
        instance = cls.from_voigt(c_ij)
        return instance.zeroed(tol)