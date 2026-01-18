from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
class Expand(BooleanOption, metaclass=OptionType):
    """``expand`` option to polynomial manipulation functions. """
    option = 'expand'
    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return True