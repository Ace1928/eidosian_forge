import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _firstfirst(iterable):
    """
    Given a nested iterable, i.e., list of lists, return the first element
    of the first non-empty element.
    """
    for i in iterable:
        for j in i:
            return j