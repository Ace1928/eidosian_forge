from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
@deprecated(last_supported_version='0.3.0', will_be_missing_in='0.8.0', use_instead=Species)
class Solute(Substance):
    """[DEPRECATED] Use `.Species` instead

    Counter-intuitive to its name Solute has an additional
    property 'precipitate'

    """

    def __init__(self, *args, **kwargs):
        precipitate = kwargs.pop('precipitate', False)
        Substance.__init__(self, *args, **kwargs)
        self.precipitate = precipitate

    @classmethod
    def from_formula(cls, formula, **kwargs):
        if formula.endswith('(s)'):
            kwargs['precipitate'] = True
        return cls(formula, latex_name=formula_to_latex(formula), unicode_name=formula_to_unicode(formula), html_name=formula_to_html(formula), composition=formula_to_composition(formula), **kwargs)