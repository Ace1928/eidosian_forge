from operator import itemgetter
from .printer import Printer
from ..units import is_quantity
def _Reaction_str(self, rxn, **kwargs):
    fmtstr = self._str('{}{}%s{}%s{}{}') % self._get('Reaction_around_arrow', **kwargs)
    return fmtstr.format(*self._Reaction_parts(rxn, **kwargs))