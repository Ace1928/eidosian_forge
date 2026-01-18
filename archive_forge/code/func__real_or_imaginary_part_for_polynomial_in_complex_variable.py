from ..sage_helper import _within_sage, sage_method, SageNotAvailable
def _real_or_imaginary_part_for_polynomial_in_complex_variable(polynomial, start):
    """
    Given a polynomial p with rational coefficients, return the
    real (start = 0) / imaginary (start = 1) part of p(x + y * I).

    The result is a sage symbolic expression in x and y with rational
    coefficients.
    """
    return sum([coeff * _real_or_imaginary_part_of_power_of_complex_number(i, start) for i, coeff in enumerate(polynomial.coefficients(sparse=False))])