from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def R13_time_vector_to_upper_halfspace(v):
    """
    Take a unit time-like vector in the 1,3-hyperboloid
    model and returns the corresponding (finite) point
    in the upper half space model
    H^3 = { x + y * i + t * j : t > 0 } as triple
    (x, y, t).
    """
    klein = [v[1] / v[0], v[2] / v[0], v[3] / v[0]]
    klein_sqr = sum([x ** 2 for x in klein])
    poincare_factor = 1.0 / (1.0 + (1.0 - klein_sqr).sqrt())
    a, b, c = [x * poincare_factor for x in klein]
    denom = (a - 1.0) ** 2 + b ** 2 + c ** 2
    return [2.0 * b / denom, 2.0 * c / denom, (1.0 - a ** 2 - b ** 2 - c ** 2) / denom]