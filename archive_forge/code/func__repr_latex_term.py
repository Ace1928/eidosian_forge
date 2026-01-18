import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@classmethod
def _repr_latex_term(cls, i, arg_str, needs_parens):
    if cls.basis_name is None:
        raise NotImplementedError('Subclasses must define either a basis name, or override _repr_latex_term(i, arg_str, needs_parens)')
    return f'{{{cls.basis_name}}}_{{{i}}}({arg_str})'