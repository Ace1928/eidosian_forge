from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
class MaterialsProjectCompatibility(CorrectionsList):
    """This class implements the GGA/GGA+U mixing scheme, which allows mixing of
    entries. Note that this should only be used for VASP calculations using the
    MaterialsProject parameters (see pymatgen.io.vasp.sets.MPVaspInputSet).
    Using this compatibility scheme on runs with different parameters is not
    valid.
    """

    def __init__(self, compat_type: str='Advanced', correct_peroxide: bool=True, check_potcar_hash: bool=False) -> None:
        """
        Args:
            compat_type: Two options, GGA or Advanced. GGA means all GGA+U
                entries are excluded. Advanced means mixing scheme is
                implemented to make entries compatible with each other,
                but entries which are supposed to be done in GGA+U will have the
                equivalent GGA entries excluded. For example, Fe oxides should
                have a U value under the Advanced scheme. A GGA Fe oxide run
                will therefore be excluded under the scheme.
            correct_peroxide: Specify whether peroxide/superoxide/ozonide
                corrections are to be applied or not.
            check_potcar_hash (bool): Use potcar hash to verify potcars are correct.
            silence_deprecation (bool): Silence deprecation warning. Defaults to False.
        """
        warnings.warn('MaterialsProjectCompatibility is deprecated, Materials Project formation energies use the newer MaterialsProject2020Compatibility scheme.', DeprecationWarning)
        self.compat_type = compat_type
        self.correct_peroxide = correct_peroxide
        self.check_potcar_hash = check_potcar_hash
        file_path = f'{MODULE_DIR}/MPCompatibility.yaml'
        super().__init__([PotcarCorrection(MPRelaxSet, check_hash=check_potcar_hash), GasCorrection(file_path), AnionCorrection(file_path, correct_peroxide=correct_peroxide), UCorrection(file_path, MPRelaxSet, compat_type)])