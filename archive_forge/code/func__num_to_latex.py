import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _num_to_latex(raw_value, decimals=15, first_term=True, coefficient=False):
    """Convert a complex number to latex code suitable for a ket expression

    Args:
        raw_value (complex): Value to convert.
        decimals (int): Number of decimal places to round to (default 15).
        coefficient (bool): Whether the number is to be used as a coefficient
                            of a ket.
        first_term (bool): If a coefficient, whether this number is the first
                           coefficient in the expression.
    Returns:
        str: latex code
    """
    import sympy
    raw_value = np.around(raw_value, decimals=decimals)
    value = sympy.nsimplify(raw_value, rational=False)
    if isinstance(value, sympy.core.numbers.Rational) and value.denominator > 50:
        value = value.evalf()
    if isinstance(value, sympy.core.numbers.Float):
        value = round(value, decimals)
    element = sympy.latex(value, full_prec=False)
    if not coefficient:
        return element
    if isinstance(value, sympy.core.Add):
        element = f'({element})'
    if element == '1':
        element = ''
    if element == '-1':
        element = '-'
    if not first_term and (not element.startswith('-')):
        element = f'+{element}'
    return element