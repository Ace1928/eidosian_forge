from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from operator import attrgetter
from .basic import Basic
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .numbers import ilcm, igcd, equal_valued
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift
from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational
@property
def _mpc_(self):
    """
        Convert self to an mpmath mpc if possible
        """
    from .numbers import Float
    re_part, rest = self.as_coeff_Add()
    im_part, imag_unit = rest.as_coeff_Mul()
    if not imag_unit == S.ImaginaryUnit:
        raise AttributeError('Cannot convert Add to mpc. Must be of the form Number + Number*I')
    return (Float(re_part)._mpf_, Float(im_part)._mpf_)