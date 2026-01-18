import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@classmethod
def _str_term_unicode(cls, i, arg_str):
    """
        String representation of single polynomial term using unicode
        characters for superscripts and subscripts.
        """
    if cls.basis_name is None:
        raise NotImplementedError('Subclasses must define either a basis_name, or override _str_term_unicode(cls, i, arg_str)')
    return f'·{cls.basis_name}{i.translate(cls._subscript_mapping)}({arg_str})'