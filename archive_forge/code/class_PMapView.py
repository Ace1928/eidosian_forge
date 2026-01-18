from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform
class PMapView:
    """View type for the persistent map/dict type `PMap`.

    Provides an equivalent of Python's built-in `dict_values` and `dict_items`
    types that result from expreessions such as `{}.values()` and
    `{}.items()`. The equivalent for `{}.keys()` is absent because the keys are
    instead represented by a `PSet` object, which can be created in `O(1)` time.

    The `PMapView` class is overloaded by the `PMapValues` and `PMapItems`
    classes which handle the specific case of values and items, respectively

    Parameters
    ----------
    m : mapping
        The mapping/dict-like object of which a view is to be created. This
        should generally be a `PMap` object.
    """

    def __init__(self, m):
        if not isinstance(m, PMap):
            if isinstance(m, Mapping):
                m = pmap(m)
            else:
                raise TypeError('PViewMap requires a Mapping object')
        object.__setattr__(self, '_map', m)

    def __len__(self):
        return len(self._map)

    def __setattr__(self, k, v):
        raise TypeError('%s is immutable' % (type(self),))

    def __reversed__(self):
        raise TypeError('Persistent maps are not reversible')