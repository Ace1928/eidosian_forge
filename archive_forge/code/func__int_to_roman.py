from __future__ import annotations
import warnings
import numpy as np
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.core import Element, Species
def _int_to_roman(number):
    """Utility method to convert an int (less than 20) to a roman numeral."""
    roman_conv = [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    result = []
    for arabic, roman in roman_conv:
        factor, number = divmod(number, arabic)
        result.append(roman * factor)
        if number == 0:
            break
    return ''.join(result)