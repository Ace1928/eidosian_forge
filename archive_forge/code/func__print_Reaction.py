from operator import itemgetter
from .printer import Printer
from ..units import is_quantity
def _print_Reaction(self, rxn, **kwargs):
    res = self._Reaction_str(rxn, **kwargs)
    if self._get('with_param', **kwargs) and rxn.param is not None:
        res += self._get('Reaction_param_separator', **kwargs)
        try:
            res += getattr(rxn.param, self._get('repr_name', **kwargs))(self._get('magnitude_fmt', **kwargs))
        except AttributeError:
            res += self._Reaction_param_str(rxn, **kwargs)
    if self._get('with_name', **kwargs) and rxn.name is not None:
        res += self._get('Reaction_param_separator', **kwargs)
        res += rxn.name
    return res