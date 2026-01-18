from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.composition import Composition
def _normalization_factor(self, mode: Literal['formula_unit', 'atom']='formula_unit') -> float:
    if mode == 'atom':
        factor = self.composition.num_atoms
    elif mode == 'formula_unit':
        factor = self.composition.get_reduced_composition_and_factor()[1]
    else:
        raise ValueError(f'{mode} is not an allowed option for normalization')
    return factor