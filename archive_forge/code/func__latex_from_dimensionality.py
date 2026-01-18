from __future__ import (absolute_import, division, print_function)
from math import log
import numpy as np
def _latex_from_dimensionality(dim):
    from quantities.markup import format_units_latex
    return format_units_latex(dim, mult='\\\\cdot')