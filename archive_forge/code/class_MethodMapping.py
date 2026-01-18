import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class MethodMapping:
    """Stores the mapping between callee and caller methods for a router.

    This class is primarily used in a ``get_metadata_routing()`` of a router
    object when defining the mapping between a sub-object (a sub-estimator or a
    scorer) to the router's methods. It stores a collection of ``Route``
    namedtuples.

    Iterating through an instance of this class will yield named
    ``MethodPair(callee, caller)`` tuples.

    .. versionadded:: 1.3
    """

    def __init__(self):
        self._routes = []

    def __iter__(self):
        return iter(self._routes)

    def add(self, *, callee, caller):
        """Add a method mapping.

        Parameters
        ----------
        callee : str
            Child object's method name. This method is called in ``caller``.

        caller : str
            Parent estimator's method name in which the ``callee`` is called.

        Returns
        -------
        self : MethodMapping
            Returns self.
        """
        if callee not in METHODS:
            raise ValueError(f'Given callee:{callee} is not a valid method. Valid methods are: {METHODS}')
        if caller not in METHODS:
            raise ValueError(f'Given caller:{caller} is not a valid method. Valid methods are: {METHODS}')
        self._routes.append(MethodPair(callee=callee, caller=caller))
        return self

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : list
            A serialized version of the instance in the form of a list.
        """
        result = list()
        for route in self._routes:
            result.append({'callee': route.callee, 'caller': route.caller})
        return result

    @classmethod
    def from_str(cls, route):
        """Construct an instance from a string.

        Parameters
        ----------
        route : str
            A string representing the mapping, it can be:

              - `"one-to-one"`: a one to one mapping for all methods.
              - `"method"`: the name of a single method, such as ``fit``,
                ``transform``, ``score``, etc.

        Returns
        -------
        obj : MethodMapping
            A :class:`~sklearn.utils.metadata_routing.MethodMapping` instance
            constructed from the given string.
        """
        routing = cls()
        if route == 'one-to-one':
            for method in METHODS:
                routing.add(callee=method, caller=method)
        elif route in METHODS:
            routing.add(callee=route, caller=route)
        else:
            raise ValueError("route should be 'one-to-one' or a single method!")
        return routing

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))