from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def _beta_tup(beta, x_unit, y_unit):
    return tuple((coeff * y_unit / x_unit ** i for i, coeff in enumerate(beta)))