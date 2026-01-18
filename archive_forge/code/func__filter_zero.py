import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _filter_zero(d):
    """
    Given a dict, filter out all items where the value is 0.
    """
    return dict(((k, v) for k, v in d.items() if v != 0))