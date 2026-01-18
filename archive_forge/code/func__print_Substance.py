from .numbers import number_to_scientific_latex
from .string import StrPrinter
from ..units import _latex_from_dimensionality
def _print_Substance(self, substance, **kwargs):
    return substance.latex_name or substance.name