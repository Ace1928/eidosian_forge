import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class MetadataRequest:
    """Contains the metadata request info of a consumer.

    Instances of `MethodMetadataRequest` are used in this class for each
    available method under `metadatarequest.{method}`.

    Consumer-only classes such as simple estimators return a serialized
    version of this class as the output of `get_metadata_routing()`.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """
    _type = 'metadata_request'

    def __init__(self, owner):
        self.owner = owner
        for method in SIMPLE_METHODS:
            setattr(self, method, MethodMetadataRequest(owner=owner, method=method))

    def consumes(self, method, params):
        """Check whether the given parameters are consumed by the given method.

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method to check.

        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by the given method.
        """
        return getattr(self, method)._consumes(params=params)

    def __getattr__(self, name):
        if name not in COMPOSITE_METHODS:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        requests = {}
        for method in COMPOSITE_METHODS[name]:
            mmr = getattr(self, method)
            existing = set(requests.keys())
            upcoming = set(mmr.requests.keys())
            common = existing & upcoming
            conflicts = [key for key in common if requests[key] != mmr._requests[key]]
            if conflicts:
                raise ValueError(f'Conflicting metadata requests for {', '.join(conflicts)} while composing the requests for {name}. Metadata with the same name for methods {', '.join(COMPOSITE_METHODS[name])} should have the same request value.')
            requests.update(mmr._requests)
        return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)

    def _get_param_names(self, method, return_alias, ignore_self_request=None):
        """Get names of all metadata that can be consumed or routed by specified             method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        ignore_self_request : bool
            Ignored. Present for API compatibility.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        return getattr(self, method)._get_param_names(return_alias=return_alias)

    def _route_params(self, *, method, params):
        """Prepare the given parameters to be passed to the method.

        The output of this method can be used directly as the input to the
        corresponding method as extra keyword arguments to pass metadata.

        Parameters
        ----------
        method : str
            The name of the method for which the parameters are requested and
            routed.

        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the
            corresponding method.
        """
        return getattr(self, method)._route_params(params=params)

    def _check_warnings(self, *, method, params):
        """Check whether metadata is passed which is marked as WARN.

        If any metadata is passed which is marked as WARN, a warning is raised.

        Parameters
        ----------
        method : str
            The name of the method for which the warnings should be checked.

        params : dict
            The metadata passed to a method.
        """
        getattr(self, method)._check_warnings(params=params)

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        output = dict()
        for method in SIMPLE_METHODS:
            mmr = getattr(self, method)
            if len(mmr.requests):
                output[method] = mmr._serialize()
        return output

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))