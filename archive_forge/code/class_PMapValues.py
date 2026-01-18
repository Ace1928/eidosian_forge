from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform
class PMapValues(PMapView):
    """View type for the values of the persistent map/dict type `PMap`.

    Provides an equivalent of Python's built-in `dict_values` type that result
    from expreessions such as `{}.values()`. See also `PMapView`.

    Parameters
    ----------
    m : mapping
        The mapping/dict-like object of which a view is to be created. This
        should generally be a `PMap` object.
    """

    def __iter__(self):
        return self._map.itervalues()

    def __contains__(self, arg):
        return arg in self._map.itervalues()

    def __str__(self):
        return f'pmap_values({list(iter(self))})'

    def __repr__(self):
        return f'pmap_values({list(iter(self))})'

    def __eq__(self, x):
        if x is self:
            return True
        else:
            return False