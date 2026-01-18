from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def _parse_chomp_and_rank(m, f, m_dict, m_points):
    """A helper method for formula parsing that helps in interpreting and
            ranking indeterminate formulas
            Author: Anubhav Jain.

            Args:
                m: A regex match, with the first group being the element and
                    the second group being the amount
                f: The formula part containing the match
                m_dict: A symbol:amt dictionary from the previously parsed
                    formula
                m_points: Number of points gained from the previously parsed
                    formula

            Returns:
                A tuple of (f, m_dict, points) where m_dict now contains data
                from the match and the match has been removed (chomped) from
                the formula f. The "goodness" of the match determines the
                number of points returned for chomping. Returns
                (None, None, None) if no element could be found...
            """
    points = 0
    points_first_capital = 100
    points_second_lowercase = 100
    el = m.group(1)
    if len(el) > 2 or len(el) < 1:
        raise ValueError('Invalid element symbol entered!')
    amt = float(m.group(2)) if m.group(2).strip() != '' else 1
    char1 = el[0]
    char2 = el[1] if len(el) > 1 else ''
    if char1 == char1.upper():
        points += points_first_capital
    if char2 and char2 == char2.lower():
        points += points_second_lowercase
    el = char1.upper() + char2.lower()
    if Element.is_valid_symbol(el):
        if el in m_dict:
            m_dict[el] += amt * factor
        else:
            m_dict[el] = amt * factor
        return (f.replace(m.group(), '', 1), m_dict, m_points + points)
    return (None, None, None)