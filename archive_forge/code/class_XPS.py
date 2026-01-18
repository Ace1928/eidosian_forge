from __future__ import annotations
import collections
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.due import Doi, due
class XPS(Spectrum):
    """Class representing an X-ray photoelectron spectra."""
    XLABEL = 'Binding Energy (eV)'
    YLABEL = 'Intensity'

    @classmethod
    def from_dos(cls, dos: CompleteDos) -> Self:
        """
        Args:
            dos: CompleteDos object with project element-orbital DOS.
            Can be obtained from Vasprun.get_complete_dos.
            sigma: Smearing for Gaussian.

        Returns:
            XPS
        """
        total = np.zeros(dos.energies.shape)
        for el in dos.structure.composition:
            spd_dos = dos.get_element_spd_dos(el)
            for orb, pdos in spd_dos.items():
                weight = CROSS_SECTIONS[el.symbol].get(str(orb))
                if weight is not None:
                    total += pdos.get_densities() * weight
                else:
                    warnings.warn(f'No cross-section for {el}{orb}')
        return XPS(-dos.energies, total / np.max(total))