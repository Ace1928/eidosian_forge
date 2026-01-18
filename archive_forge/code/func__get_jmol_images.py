import collections
from math import sin, pi, sqrt
from numbers import Real, Integral
from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.atoms import Atoms
import ase.units as units
import ase.io
from ase.utils import jsonable, lazymethod
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spectrum.dosdata import RawDOSData
from ase.spectrum.doscollection import DOSCollection
@staticmethod
def _get_jmol_images(atoms: Atoms, energies: np.ndarray, modes: np.ndarray, ir_intensities: Union[Sequence[float], np.ndarray]=None) -> Iterator[Atoms]:
    """Get vibrational modes as a series of Atoms with attached data

        For each image (Atoms object):

            - eigenvalues are attached to image.arrays['mode']
            - "mode#" and "frequency_cm-1" are set in image.info
            - "IR_intensity" is set if provided in ir_intensities
            - "masses" is removed

        This is intended to set up the object for JMOL-compatible export using
        ase.io.extxyz.


        Args:
            atoms: The base atoms object; all images have the same positions
            energies: Complex vibrational energies in eV
            modes: Eigenvectors array corresponding to atoms and energies. This
                should cover the full set of atoms (i.e. modes =
                vib.get_modes(all_atoms=True)).
            ir_intensities: If available, IR intensities can be included in the
                header lines. This does not affect the visualisation, but may
                be convenient when comparing to experimental data.
        Returns:
            Iterator of Atoms objects

        """
    for i, (energy, mode) in enumerate(zip(energies, modes)):
        if energy.imag > energy.real:
            energy = float(-energy.imag)
        else:
            energy = energy.real
        image = atoms.copy()
        image.info.update({'mode#': str(i), 'frequency_cm-1': energy / units.invcm})
        image.arrays['mode'] = mode
        if image.has('masses'):
            del image.arrays['masses']
        if ir_intensities is not None:
            image.info['IR_intensity'] = float(ir_intensities[i])
        yield image