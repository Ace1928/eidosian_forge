from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
@classmethod
def _str_from_formulas(cls, coeffs, formulas) -> str:
    reactant_str = []
    product_str = []
    for amt, formula in zip(coeffs, formulas):
        if abs(amt + 1) < cls.TOLERANCE:
            reactant_str.append(formula)
        elif abs(amt - 1) < cls.TOLERANCE:
            product_str.append(formula)
        elif amt < -cls.TOLERANCE:
            reactant_str.append(f'{-amt:.4g} {formula}')
        elif amt > cls.TOLERANCE:
            product_str.append(f'{amt:.4g} {formula}')
    return f'{' + '.join(reactant_str)} -> {' + '.join(product_str)}'