from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
class OptionType(type):
    """Base type for all options that does registers options. """

    def __init__(cls, *args, **kwargs):

        @property
        def getter(self):
            try:
                return self[cls.option]
            except KeyError:
                return cls.default()
        setattr(Options, cls.option, getter)
        Options.__options__[cls.option] = cls