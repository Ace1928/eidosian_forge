from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
@staticmethod
def _sanitize_input_structure(struct: Structure) -> Structure:
    """Sanitize our input structure by removing magnetic information
        and making primitive.

        Args:
            struct: Structure

        Returns:
            Structure
        """
    struct = struct.copy()
    struct.remove_spin()
    struct = struct.get_primitive_structure(use_site_props=False)
    if 'magmom' in struct.site_properties:
        struct.remove_site_property('magmom')
    return struct