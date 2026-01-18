from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
class Gen(Flag, metaclass=OptionType):
    """``gen`` flag to polynomial manipulation functions. """
    option = 'gen'

    @classmethod
    def default(cls):
        return 0

    @classmethod
    def preprocess(cls, gen):
        if isinstance(gen, (Basic, int)):
            return gen
        else:
            raise OptionError("invalid argument for 'gen' option")