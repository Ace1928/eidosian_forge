from __future__ import annotations
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import polar
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, fast_norm
from pymatgen.core.interface import Interface, label_termination
from pymatgen.core.surface import SlabGenerator
Generates interface structures given the film and substrate structure
        as well as the desired terminations.

        Args:
            termination (tuple[str, str]): termination from self.termination list
            gap (float, optional): gap between film and substrate. Defaults to 2.0.
            vacuum_over_film (float, optional): vacuum over the top of the film. Defaults to 20.0.
            film_thickness (float, optional): the film thickness. Defaults to 1.
            substrate_thickness (float, optional): substrate thickness. Defaults to 1.
            in_layers (bool, optional): set the thickness in layer units. Defaults to True.

        Yields:
            Iterator[Interface]: interfaces from slabs
        