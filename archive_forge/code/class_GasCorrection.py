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
class GasCorrection(Correction):
    """Correct gas energies to obtain the right formation energies. Note that
    this depends on calculations being run within the same input set.
    Used by legacy MaterialsProjectCompatibility and MITCompatibility.
    """

    def __init__(self, config_file):
        """
        Args:
            config_file: Path to the selected compatibility.yaml config file.
        """
        config = loadfn(config_file)
        self.name = config['Name']
        self.cpd_energies = config['Advanced']['CompoundEnergies']

    def get_correction(self, entry) -> ufloat:
        """
        Args:
            entry: A ComputedEntry/ComputedStructureEntry.

        Returns:
            Correction.
        """
        comp = entry.composition
        correction = ufloat(0.0, 0.0)
        rform = entry.reduced_formula
        if rform in self.cpd_energies:
            correction += self.cpd_energies[rform] * comp.num_atoms - entry.uncorrected_energy
        return correction

    def __str__(self):
        return f'{self.name} Gas Correction'