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
@cached_class
class AqueousCorrection(Correction):
    """This class implements aqueous phase compound corrections for elements
    and H2O.

    Used only by MITAqueousCompatibility.
    """

    def __init__(self, config_file, error_file=None):
        """
        Args:
            config_file: Path to the selected compatibility.yaml config file.
            error_file: Path to the selected compatibilityErrors.yaml config file.
        """
        config = loadfn(config_file)
        self.cpd_energies = config['AqueousCompoundEnergies']
        self.comp_correction = config.get('CompositionCorrections', defaultdict(float))
        self.oxide_correction = config.get('OxideCorrections', defaultdict(float))
        self.name = config['Name']
        if error_file:
            e = loadfn(error_file)
            self.cpd_errors = e.get('AqueousCompoundEnergies', defaultdict(float))
        else:
            self.cpd_errors = defaultdict(float)

    def get_correction(self, entry) -> ufloat:
        """
        Args:
            entry: A ComputedEntry/ComputedStructureEntry.

        Returns:
            Correction, Uncertainty.
        """
        comp = entry.composition
        rform = comp.reduced_formula
        cpd_energies = self.cpd_energies
        correction = ufloat(0.0, 0.0)
        if rform in cpd_energies:
            if rform in ['H2', 'H2O']:
                corr = cpd_energies[rform] * comp.num_atoms - entry.uncorrected_energy - entry.correction
                err = self.cpd_errors[rform] * comp.num_atoms
                correction += ufloat(corr, err)
            else:
                corr = cpd_energies[rform] * comp.num_atoms
                err = self.cpd_errors[rform] * comp.num_atoms
                correction += ufloat(corr, err)
        if rform != 'H2O':
            nH2O = int(min(comp['H'] / 2.0, comp['O']))
            if nH2O > 0:
                correction -= ufloat((comp['H'] - nH2O / 2) * self.comp_correction['H'], 0.0)
                correction -= ufloat((comp['O'] - nH2O) * (self.comp_correction['oxide'] + self.oxide_correction['oxide']), 0.0)
                correction += ufloat(-1 * MU_H2O * nH2O, 0.0)
        return correction

    def __str__(self):
        return f'{self.name} Aqueous Correction'