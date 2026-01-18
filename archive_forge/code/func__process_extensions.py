from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def _process_extensions(extensions):
    """
    Given a tower of field extensions, find the number field defining
    polynomial and write all variables in terms of the root in that
    polynomial.
    """
    if not extensions:
        return (None, {})
    poly, var, degree = extensions[0]
    ext_assignments = {var: Polynomial.from_variable_name('x')}
    number_field = poly.substitute(ext_assignments)
    for extension in extensions[1:]:
        poly, var, degree = extension
        poly = poly.substitute(ext_assignments)
        number_field, old_x_in_new_x, k = my_rnfequation(number_field, poly)
        ext_assignments = dict([(key, poly.substitute({'x': old_x_in_new_x})) for key, poly in ext_assignments.items()])
        ext_assignments[var] = Polynomial.from_variable_name('x') - Polynomial.constant_polynomial(k) * old_x_in_new_x
    return (number_field, ext_assignments)