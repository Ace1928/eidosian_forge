from operator import itemgetter
from .printer import Printer
from ..units import is_quantity
def _Reaction_param_str(self, rxn, **kwargs):
    mag_fmt = self._get('magnitude_fmt', **kwargs)
    unit_fmt = self._get('unit_fmt', **kwargs)
    try:
        magnitude_str = mag_fmt(rxn.param.magnitude)
        unit_str = unit_fmt(rxn.param.dimensionality)
    except AttributeError:
        if is_quantity(rxn.param) or isinstance(rxn.param, (float,)):
            return mag_fmt(rxn.param)
        else:
            return str(rxn.param)
    else:
        return magnitude_str + self._str(' ') + unit_str