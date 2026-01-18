from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1021/acs.jpclett.0c02405'), description='AccurAccurate and Numerically Efficient r2SCAN Meta-Generalized Gradient Approximation')
@due.dcite(Doi('10.1103/PhysRevLett.115.036402'), description='Strongly Constrained and Appropriately Normed Semilocal Density Functional')
@due.dcite(Doi('10.1103/PhysRevB.93.155109'), description='Efficient generation of generalized Monkhorst-Pack grids through the use of informatics')
@dataclass
class MPScanRelaxSet(DictSet):
    """
    Class for writing a relaxation input set using the accurate and numerically
    efficient r2SCAN variant of the Strongly Constrained and Appropriately Normed
    (SCAN) metaGGA density functional.

    Notes:
        1. This functional is officially supported in VASP 6.0.0 and above. On older version,
        source code may be obtained by contacting the authors of the referenced manuscript.
        The original SCAN functional, available from VASP 5.4.3 onwards, maybe used instead
        by passing `user_incar_settings={"METAGGA": "SCAN"}` when instantiating this InputSet.
        r2SCAN and SCAN are expected to yield very similar results.

        2. Meta-GGA calculations require POTCAR files that include
        information on the kinetic energy density of the core-electrons,
        i.e. "PBE_52" or "PBE_54". Make sure the POTCARs include the
        following lines (see VASP wiki for more details):

            $ grep kinetic POTCAR
            kinetic energy-density
            mkinetic energy-density pseudized
            kinetic energy density (partial)

    Args:
        bandgap (float): Bandgap of the structure in eV. The bandgap is used to
            compute the appropriate k-point density and determine the
            smearing settings.
            Metallic systems (default, bandgap = 0) use a KSPACING value of 0.22
            and Methfessel-Paxton order 2 smearing (ISMEAR=2, SIGMA=0.2).
            Non-metallic systems (bandgap > 0) use the tetrahedron smearing
            method (ISMEAR=-5, SIGMA=0.05). The KSPACING value is
            calculated from the bandgap via Eqs. 25 and 29 of Wisesa, McGill,
            and Mueller [1] (see References). Note that if 'user_incar_settings'
            or 'user_kpoints_settings' override KSPACING, the calculation from
            bandgap is not performed.
        vdw (str): set "rVV10" to enable SCAN+rVV10, which is a versatile
            van der Waals density functional by combing the SCAN functional
            with the rVV10 non-local correlation functional. rvv10 is the only
            dispersion correction available for SCAN at this time.
        **kwargs: Same as those supported by DictSet.

        References:
            [1] P. Wisesa, K.A. McGill, T. Mueller, Efficient generation of
            generalized Monkhorst-Pack grids through the use of informatics,
            Phys. Rev. B. 93 (2016) 1-10. doi:10.1103/PhysRevB.93.155109.

    References:
        James W. Furness, Aaron D. Kaplan, Jinliang Ning, John P. Perdew, and Jianwei Sun.
        Accurate and Numerically Efficient r2SCAN Meta-Generalized Gradient Approximation.
        The Journal of Physical Chemistry Letters 0, 11 DOI: 10.1021/acs.jpclett.0c02405
    """
    bandgap: float | None = None
    user_potcar_functional: UserPotcarFunctional = 'PBE_54'
    auto_ismear: bool = True
    CONFIG = _load_yaml_config('MPSCANRelaxSet')
    _valid_potcars = ('PBE_52', 'PBE_54')

    def __post_init__(self):
        super().__post_init__()
        if self.vdw and self.vdw != 'rvv10':
            warnings.warn('Use of van der waals functionals other than rVV10 with SCAN is not supported at this time. ')
            vdw_par = loadfn(str(MODULE_DIR / 'vdW_parameters.yaml'))
            for k in vdw_par[self.vdw]:
                self._config_dict['INCAR'].pop(k, None)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        return {'KSPACING': 'auto'}