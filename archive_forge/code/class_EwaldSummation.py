from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1016/0010-4655(96)00016-1'), description='Ewald summation techniques in perspective: a survey', path='pymatgen.analysis.ewald.EwaldSummation')
class EwaldSummation(MSONable):
    """
    Calculates the electrostatic energy of a periodic array of charges using
    the Ewald technique.

    Ref:
        Ewald summation techniques in perspective: a survey
        Abdulnour Y. Toukmaji and John A. Board Jr.
        DOI: 10.1016/0010-4655(96)00016-1
        URL: http://www.ee.duke.edu/~ayt/ewaldpaper/ewaldpaper.html

    This matrix can be used to do fast calculations of Ewald sums after species
    removal.

    E = E_recip + E_real + E_point

    Atomic units used in the code, then converted to eV.
    """
    CONV_FACT = 10000000000.0 * constants.e / (4 * pi * constants.epsilon_0)

    def __init__(self, structure, real_space_cut=None, recip_space_cut=None, eta=None, acc_factor=12.0, w=1 / 2 ** 0.5, compute_forces=False):
        """
        Initializes and calculates the Ewald sum. Default convergence
        parameters have been specified, but you can override them if you wish.

        Args:
            structure (Structure): Input structure that must have proper
                Species on all sites, i.e. Element with oxidation state. Use
                Structure.add_oxidation_state... for example.
            real_space_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum. Defaults to None,
                which means determine automatically using the formula given
                in gulp 3.1 documentation.
            recip_space_cut (float): Reciprocal space cutoff radius.
                Defaults to None, which means determine automatically using
                the formula given in gulp 3.1 documentation.
            eta (float): The screening parameter. Defaults to None, which means
                determine automatically.
            acc_factor (float): No. of significant figures each sum is
                converged to.
            w (float): Weight parameter, w, has been included that represents
                the relative computational expense of calculating a term in
                real and reciprocal space. Default of 0.7 reproduces result
                similar to GULP 4.2. This has little effect on the total
                energy, but may influence speed of computation in large
                systems. Note that this parameter is used only when the
                cutoffs are set to None.
            compute_forces (bool): Whether to compute forces. False by
                default since it is usually not needed.
        """
        self._struct = structure
        self._charged = abs(structure.charge) > 1e-08
        self._vol = structure.volume
        self._compute_forces = compute_forces
        self._acc_factor = acc_factor
        self._eta = eta or (len(structure) * w / self._vol ** 2) ** (1 / 3) * pi
        self._sqrt_eta = sqrt(self._eta)
        self._accf = sqrt(log(10 ** acc_factor))
        self._rmax = real_space_cut or self._accf / self._sqrt_eta
        self._gmax = recip_space_cut or 2 * self._sqrt_eta * self._accf
        self._oxi_states = [compute_average_oxidation_state(site) for site in structure]
        self._coords = np.array(self._struct.cart_coords)
        self._initialized = False
        self._recip = self._real = self._point = self._forces = None
        self._charged_cell_energy = -EwaldSummation.CONV_FACT / 2 * np.pi / structure.volume / self._eta * structure.charge ** 2

    def compute_partial_energy(self, removed_indices):
        """
        Gives total Ewald energy for certain sites being removed, i.e. zeroed
        out.
        """
        total_energy_matrix = self.total_energy_matrix.copy()
        for idx in removed_indices:
            total_energy_matrix[idx, :] = 0
            total_energy_matrix[:, idx] = 0
        return sum(sum(total_energy_matrix))

    def compute_sub_structure(self, sub_structure, tol: float=0.001):
        """
        Gives total Ewald energy for an sub structure in the same
        lattice. The sub_structure must be a subset of the original
        structure, with possible different charges.

        Args:
            substructure (Structure): Substructure to compute Ewald sum for.
            tol (float): Tolerance for site matching in fractional coordinates.

        Returns:
            Ewald sum of substructure.
        """
        total_energy_matrix = self.total_energy_matrix.copy()

        def find_match(site):
            for test_site in sub_structure:
                frac_diff = abs(np.array(site.frac_coords) - np.array(test_site.frac_coords)) % 1
                frac_diff = [abs(a) < tol or abs(a) > 1 - tol for a in frac_diff]
                if all(frac_diff):
                    return test_site
            return None
        matches = []
        for idx, site in enumerate(self._struct):
            matching_site = find_match(site)
            if matching_site:
                new_charge = compute_average_oxidation_state(matching_site)
                old_charge = self._oxi_states[idx]
                scaling_factor = new_charge / old_charge
                matches.append(matching_site)
            else:
                scaling_factor = 0
            total_energy_matrix[idx, :] *= scaling_factor
            total_energy_matrix[:, idx] *= scaling_factor
        if len(matches) != len(sub_structure):
            output = ['Missing sites.']
            for site in sub_structure:
                if site not in matches:
                    output.append(f'unmatched = {site}')
            raise ValueError('\n'.join(output))
        return sum(sum(total_energy_matrix))

    @property
    def reciprocal_space_energy(self):
        """The reciprocal space energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(sum(self._recip))

    @property
    def reciprocal_space_energy_matrix(self):
        """
        The reciprocal space energy matrix. Each matrix element (i, j)
        corresponds to the interaction energy between site i and site j in
        reciprocal space.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self._recip

    @property
    def real_space_energy(self):
        """The real space energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(sum(self._real))

    @property
    def real_space_energy_matrix(self):
        """
        The real space energy matrix. Each matrix element (i, j) corresponds to
        the interaction energy between site i and site j in real space.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self._real

    @property
    def point_energy(self):
        """The point energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(self._point)

    @property
    def point_energy_matrix(self):
        """
        The point space matrix. A diagonal matrix with the point terms for each
        site in the diagonal elements.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self._point

    @property
    def total_energy(self):
        """The total energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(sum(self._recip)) + sum(sum(self._real)) + sum(self._point) + self._charged_cell_energy

    @property
    def total_energy_matrix(self):
        """
        The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy between site i and site j.

        Note that this does not include the charged-cell energy, which is only important
        when the simulation cell is not charge balanced.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        total_energy = self._recip + self._real
        for idx, energy in enumerate(self._point):
            total_energy[idx, idx] += energy
        return total_energy

    @property
    def forces(self):
        """
        The forces on each site as a Nx3 matrix. Each row corresponds to a
        site.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        if not self._compute_forces:
            raise AttributeError('Forces are available only if compute_forces is True!')
        return self._forces

    def get_site_energy(self, site_index):
        """Compute the energy for a single site in the structure.

        Args:
            site_index (int): Index of site

        Returns:
            float: Energy of that site
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        if self._charged:
            warn('Per atom energies for charged structures not supported in EwaldSummation')
        return np.sum(self._recip[:, site_index]) + np.sum(self._real[:, site_index]) + self._point[site_index]

    def _calc_ewald_terms(self):
        """Calculates and sets all Ewald terms (point, real and reciprocal)."""
        self._recip, recip_forces = self._calc_recip()
        self._real, self._point, real_point_forces = self._calc_real_and_point()
        if self._compute_forces:
            self._forces = recip_forces + real_point_forces

    def _calc_recip(self):
        """
        Perform the reciprocal space summation. Calculates the quantity
        E_recip = 1/(2PiV) sum_{G < Gmax} exp(-(G.G/4/eta))/(G.G) S(G)S(-G)
        where
        S(G) = sum_{k=1,N} q_k exp(-i G.r_k)
        S(G)S(-G) = |S(G)|**2.

        This method is heavily vectorized to utilize numpy's C backend for speed.
        """
        n_sites = len(self._struct)
        prefactor = 2 * pi / self._vol
        e_recip = np.zeros((n_sites, n_sites), dtype=np.float64)
        forces = np.zeros((n_sites, 3), dtype=np.float64)
        coords = self._coords
        rcp_latt = self._struct.lattice.reciprocal_lattice
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self._gmax)
        frac_coords = [frac_coords for frac_coords, dist, _idx, _img in recip_nn if dist != 0]
        gs = rcp_latt.get_cartesian_coords(frac_coords)
        g2s = np.sum(gs ** 2, 1)
        exp_vals = np.exp(-g2s / (4 * self._eta))
        grs = np.sum(gs[:, None] * coords[None, :], 2)
        oxi_states = np.array(self._oxi_states)
        qi_qj = oxi_states[None, :] * oxi_states[:, None]
        s_reals = np.sum(oxi_states[None, :] * np.cos(grs), 1)
        s_imags = np.sum(oxi_states[None, :] * np.sin(grs), 1)
        for g, g2, gr, exp_val, s_real, s_imag in zip(gs, g2s, grs, exp_vals, s_reals, s_imags):
            m = gr[None, :] + pi / 4 - gr[:, None]
            np.sin(m, m)
            m *= exp_val / g2
            e_recip += m
            if self._compute_forces:
                pref = 2 * exp_val / g2 * oxi_states
                factor = prefactor * pref * (s_real * np.sin(gr) - s_imag * np.cos(gr))
                forces += factor[:, None] * g[None, :]
        forces *= EwaldSummation.CONV_FACT
        e_recip *= prefactor * EwaldSummation.CONV_FACT * qi_qj * 2 ** 0.5
        return (e_recip, forces)

    def _calc_real_and_point(self):
        """Determines the self energy -(eta/pi)**(1/2) * sum_{i=1}^{N} q_i**2."""
        frac_coords = self._struct.frac_coords
        force_pf = 2 * self._sqrt_eta / sqrt(pi)
        coords = self._coords
        n_sites = len(self._struct)
        e_real = np.empty((n_sites, n_sites), dtype=np.float64)
        forces = np.zeros((n_sites, 3), dtype=np.float64)
        qs = np.array(self._oxi_states)
        e_point = -qs ** 2 * sqrt(self._eta / pi)
        for idx in range(n_sites):
            nf_coords, rij, js, _ = self._struct.lattice.get_points_in_sphere(frac_coords, coords[idx], self._rmax, zip_results=False)
            inds = rij > 1e-08
            js = js[inds]
            rij = rij[inds]
            nf_coords = nf_coords[inds]
            qi = qs[idx]
            qj = qs[js]
            erfc_val = erfc(self._sqrt_eta * rij)
            new_ereals = erfc_val * qi * qj / rij
            for key in range(n_sites):
                e_real[key, idx] = np.sum(new_ereals[js == key])
            if self._compute_forces:
                nc_coords = self._struct.lattice.get_cartesian_coords(nf_coords)
                fijpf = qj / rij ** 3 * (erfc_val + force_pf * rij * np.exp(-self._eta * rij ** 2))
                forces[idx] += np.sum(np.expand_dims(fijpf, 1) * (np.array([coords[idx]]) - nc_coords) * qi * EwaldSummation.CONV_FACT, axis=0)
        e_real *= 0.5 * EwaldSummation.CONV_FACT
        e_point *= EwaldSummation.CONV_FACT
        return (e_real, e_point, forces)

    @property
    def eta(self):
        """Returns: eta value used in Ewald summation."""
        return self._eta

    def __str__(self):
        output = [f'Real = {self.real_space_energy}', f'Reciprocal = {self.reciprocal_space_energy}', f'Point = {self.point_energy}', f'Total = {self.total_energy}', f'Forces:\n{self.forces}' if self._compute_forces else 'Forces were not computed']
        return '\n'.join(output)

    def as_dict(self, verbosity: int=0) -> dict:
        """
        JSON-serialization dict representation of EwaldSummation.

        Args:
            verbosity (int): Verbosity level. Default of 0 only includes the
                matrix representation. Set to 1 for more details.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': self._struct.as_dict(), 'compute_forces': self._compute_forces, 'eta': self._eta, 'acc_factor': self._acc_factor, 'real_space_cut': self._rmax, 'recip_space_cut': self._gmax, '_recip': None if self._recip is None else self._recip.tolist(), '_real': None if self._real is None else self._real.tolist(), '_point': None if self._point is None else self._point.tolist(), '_forces': None if self._forces is None else self._forces.tolist()}

    @classmethod
    def from_dict(cls, dct: dict[str, Any], fmt: str | None=None, **kwargs) -> Self:
        """Create an EwaldSummation instance from JSON-serialized dictionary.

        Args:
            dct (dict): Dictionary representation
            fmt (str, optional): Unused. Defaults to None.

        Returns:
            EwaldSummation: class instance
        """
        summation = cls(structure=Structure.from_dict(dct['structure']), real_space_cut=dct['real_space_cut'], recip_space_cut=dct['recip_space_cut'], eta=dct['eta'], acc_factor=dct['acc_factor'], compute_forces=dct['compute_forces'])
        if dct['_recip'] is not None:
            summation._recip = np.array(dct['_recip'])
            summation._real = np.array(dct['_real'])
            summation._point = np.array(dct['_point'])
            summation._forces = np.array(dct['_forces'])
            summation._initialized = True
        return summation