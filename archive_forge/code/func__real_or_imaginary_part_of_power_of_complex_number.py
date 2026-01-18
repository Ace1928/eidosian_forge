from ..sage_helper import _within_sage, sage_method, SageNotAvailable
def _real_or_imaginary_part_of_power_of_complex_number(n, start):
    """
    Let z = x + y * I.
    If start = 0, return Re(z^n). If start = 1, return Im(z^n).
    The result is a sage symbolic expression in x and y with rational
    coefficients.
    """
    return sum([binomial(n, i) * (-1) ** (i // 2) * var('x') ** (n - i) * var('y') ** i for i in range(start, n + 1, 2)])