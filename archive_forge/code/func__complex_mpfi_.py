import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _complex_mpfi_(self, CIF):
    """
        Convert to complex interval in given ComplexIntervalField instance.
        """
    RIF = CIF(0).real().parent()
    return CIF(RIF(self._real), RIF(self._imag))