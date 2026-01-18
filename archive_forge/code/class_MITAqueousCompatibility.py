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
class MITAqueousCompatibility(CorrectionsList):
    """This class implements the GGA/GGA+U mixing scheme, which allows mixing of
    entries. Note that this should only be used for VASP calculations using the
    MIT parameters (see pymatgen.io.vasp.sets MITVaspInputSet). Using
    this compatibility scheme on runs with different parameters is not valid.
    """

    def __init__(self, compat_type: Literal['GGA', 'Advanced']='Advanced', correct_peroxide: bool=True, check_potcar_hash: bool=False) -> None:
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
        """
        self.compat_type = compat_type
        self.correct_peroxide = correct_peroxide
        self.check_potcar_hash = check_potcar_hash
        fp = f'{MODULE_DIR}/MITCompatibility.yaml'
        super().__init__([PotcarCorrection(MITRelaxSet, check_hash=check_potcar_hash), GasCorrection(fp), AnionCorrection(fp, correct_peroxide=correct_peroxide), UCorrection(fp, MITRelaxSet, compat_type), AqueousCorrection(fp)])