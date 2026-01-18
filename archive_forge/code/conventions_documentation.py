import re
from collections.abc import Iterable
from sympy.core.function import Derivative
Return whether a partial derivative symbol is required for printing

    This requires checking how many free variables there are,
    filtering out the ones that are integers. Some expressions do not have
    free variables. In that case, check its variable list explicitly to
    get the context of the expression.
    