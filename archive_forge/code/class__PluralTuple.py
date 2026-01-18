from __future__ import annotations
from operator import itemgetter
from babel.core import Locale, default_locale
class _PluralTuple(tuple):
    """A tuple with plural information."""
    __slots__ = ()
    num_plurals = property(itemgetter(0), doc='\n    The number of plurals used by the locale.')
    plural_expr = property(itemgetter(1), doc='\n    The plural expression used by the locale.')
    plural_forms = property(lambda x: 'nplurals={}; plural={};'.format(*x), doc='\n    The plural expression used by the catalog or locale.')

    def __str__(self) -> str:
        return self.plural_forms