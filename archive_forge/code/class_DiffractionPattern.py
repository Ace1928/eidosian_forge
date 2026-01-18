from __future__ import annotations
import abc
import collections
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.plotting import add_fig_kwargs, pretty_plot
class DiffractionPattern(Spectrum):
    """A representation of a diffraction pattern."""
    XLABEL = '$2\\Theta$'
    YLABEL = 'Intensity'

    def __init__(self, x, y, hkls, d_hkls):
        """
        Args:
            x: Two theta angles.
            y: Intensities
            hkls: [{"hkl": (h, k, l), "multiplicity": mult}],
                where {"hkl": (h, k, l), "multiplicity": mult}
                is a dict of Miller
                indices for all diffracted lattice facets contributing to each
                intensity.
            d_hkls: List of interplanar spacings.
        """
        super().__init__(x, y, hkls, d_hkls)
        self.hkls = hkls
        self.d_hkls = d_hkls