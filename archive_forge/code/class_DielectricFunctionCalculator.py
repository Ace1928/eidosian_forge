from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
@dataclass
class DielectricFunctionCalculator(MSONable):
    """Class for postprocessing VASP optical properties calculations.

    This objects helps load the different parameters from the vasprun.xml file but allows users to override
    them as needed.

    The standard vasprun.xml from an ``LOPTICS=.True.`` calculation already contains
    the complex frequency dependent dielectric functions.  However you have no way to decompose
    the different contributions.  Since the ``WAVEDER`` file is also written during an optical calculation,
    you can reconstruct the dielectric functions purely in Python and have full control over contribution
    from different bands and k-points.

    VASP's linear optics follow these steps:
        - Calculate the imaginary part
        - Perform symmetry operations (this is not implemented here)
        - Calculate the real part

    Currently, this Calculator only works for ``ISYM=0`` calculations since we cannot guarantee that our
    externally defined symmetry operations are the same as VASP's. This can be fixed by printing the
    symmetry operators into the vasprun.xml file. If this happens in future versions of VASP,
    we can dramatically speed up the calculations here by considering only the irreducible kpoints.
    """
    cder_real: NDArray
    cder_imag: NDArray
    eigs: NDArray
    kweights: NDArray
    nedos: int
    deltae: float
    ismear: int
    sigma: float
    efermi: float
    cshift: float
    ispin: int
    volume: float

    @classmethod
    def from_vasp_objects(cls, vrun: Vasprun, waveder: Waveder) -> Self:
        """Construct a DielectricFunction from Vasprun, Kpoint, and Waveder objects.

        Args:
            vrun: Vasprun object
            kpoint: Kpoint object
            waveder: Waveder object
        """
        bands = vrun.eigenvalues
        sspins = [Spin.up, Spin.down]
        eigs = np.stack([bands[spin] for spin in sspins[:vrun.parameters['ISPIN']]], axis=2)[..., 0]
        eigs = np.swapaxes(eigs, 0, 1)
        kweights = vrun.actual_kpoints_weights
        nedos = vrun.parameters['NEDOS']
        deltae = vrun.dielectric[0][1]
        ismear = vrun.parameters['ISMEAR']
        sigma = vrun.parameters['SIGMA']
        cshift = vrun.parameters['CSHIFT']
        efermi = vrun.efermi
        ispin = vrun.parameters['ISPIN']
        volume = vrun.final_structure.volume
        if vrun.parameters['ISYM'] != 0:
            raise NotImplementedError('ISYM != 0 is not implemented yet')
        return cls(cder_real=waveder.cder_real, cder_imag=waveder.cder_imag, eigs=eigs, kweights=kweights, nedos=nedos, deltae=deltae, ismear=ismear, sigma=sigma, efermi=efermi, cshift=cshift, ispin=ispin, volume=volume)

    @classmethod
    def from_directory(cls, directory: Path | str) -> Self:
        """Construct a DielectricFunction from a directory containing vasprun.xml and WAVEDER files."""

        def _try_reading(dtypes):
            """Return None if failed."""
            for dtype in dtypes:
                try:
                    return Waveder.from_binary(f'{directory}/WAVEDER', data_type=dtype)
                except ValueError as exc:
                    if 'reshape' in str(exc):
                        continue
                    raise exc
            return None
        vrun = Vasprun(f'{directory}/vasprun.xml')
        if 'gamma' in vrun.generator['subversion'].lower():
            waveder = _try_reading(['float64', 'float32'])
        else:
            waveder = _try_reading(['complex128', 'complex64'])
        return cls.from_vasp_objects(vrun, waveder)

    @property
    def cder(self):
        """Complex CDER from WAVEDER."""
        return self.cder_real + self.cder_imag * 1j

    def get_epsilon(self, idir: int, jdir: int, efermi: float | None=None, nedos: int | None=None, deltae: float | None=None, ismear: int | None=None, sigma: float | None=None, cshift: float | None=None, mask: NDArray | None=None) -> tuple[NDArray, NDArray]:
        """Compute the frequency dependent dielectric function.

        Args:
            idir: First direction of the dielectric tensor
            jdir: Second direction of the dielectric tensor
            efermi: Fermi energy
            nedos: Number of points in the DOS
            deltae: Energy step in the DOS
            ismear: Smearing method (only has 0:gaussian, >0:Methfessel-Paxton)
            sigma: Smearing width
            cshift: Complex shift used for Kramer-Kronig transformation
            mask: Mask for the bands/kpoint/spin index to include in the calculation
        """

        def _use_default(param, default):
            return param if param is not None else default
        efermi = _use_default(efermi, self.efermi)
        nedos = _use_default(nedos, self.nedos)
        deltae = _use_default(deltae, self.deltae)
        ismear = _use_default(ismear, self.ismear)
        sigma = _use_default(sigma, self.sigma)
        cshift = _use_default(cshift, self.cshift)
        egrid, eps_imag = epsilon_imag(cder=self.cder, eigs=self.eigs, kweights=self.kweights, efermi=efermi, nedos=nedos, deltae=deltae, ismear=ismear, sigma=sigma, idir=idir, jdir=jdir, mask=mask)
        eps_in = eps_imag * edeps * np.pi / self.volume
        eps = kramers_kronig(eps_in, nedos=nedos, deltae=deltae, cshift=cshift)
        if idir == jdir:
            eps += 1.0 + 0j
        return (egrid, eps)

    def plot_weighted_transition_data(self, idir: int, jdir: int, mask: NDArray | None=None, min_val: float=0.0):
        """Data for plotting the weight matrix elements as a scatter plot.

        Since the computation of the final spectrum (especially the smearing part)
        is still fairly expensive.  This function can be used to check the values
        of some portion of the spectrum (defined by the mask).
        In a sense, we are lookin at the imaginary part of the dielectric function
        before the smearing is applied.

        Args:
            idir: First direction of the dielectric tensor.
            jdir: Second direction of the dielectric tensor.
            mask: Mask to apply to the CDER for the bands/kpoint/spin
                index to include in the calculation
            min_val: Minimum value below this value the matrix element will not be shown.
        """
        cderm = self.cder * mask if mask is not None else self.cder
        norm_kweights = np.array(self.kweights) / np.sum(self.kweights)
        eigs_shifted = self.eigs - self.efermi
        rspin = 3 - cderm.shape[3]
        try:
            min_band0, max_band0 = (np.min(np.where(cderm)[0]), np.max(np.where(cderm)[0]))
            min_band1, max_band1 = (np.min(np.where(cderm)[1]), np.max(np.where(cderm)[1]))
        except ValueError as exc:
            if 'zero-size array' in str(exc):
                raise ValueError('No matrix elements found. Check the mask.')
            raise
        x_val = []
        y_val = []
        text = []
        _, _, nk, nspin = cderm.shape[:4]
        iter_idx = [range(min_band0, max_band0 + 1), range(min_band1, max_band1 + 1), range(nk), range(nspin)]
        num_ = (max_band0 - min_band0) * (max_band1 - min_band1) * nk * nspin
        for ib, jb, ik, ispin in tqdm(itertools.product(*iter_idx), total=num_):
            fermi_w_i = step_func(eigs_shifted[ib, ik, ispin] / self.sigma, self.ismear)
            fermi_w_j = step_func(eigs_shifted[jb, ik, ispin] / self.sigma, self.ismear)
            weight = (fermi_w_j - fermi_w_i) * rspin * norm_kweights[ik]
            A = cderm[ib, jb, ik, ispin, idir] * np.conjugate(cderm[ib, jb, ik, ispin, jdir])
            decel = self.eigs[jb, ik, ispin] - self.eigs[ib, ik, ispin]
            matrix_el = np.abs(A) * float(weight)
            if matrix_el > min_val:
                x_val.append(decel)
                y_val.append(matrix_el)
                text.append(f's:{ispin}, k:{ik}, {ib} -> {jb} ({decel:.2f})')
        return (x_val, y_val, text)