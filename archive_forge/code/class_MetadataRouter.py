import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class MetadataRouter:
    """Stores and handles metadata routing for a router object.

    This class is used by router objects to store and handle metadata routing.
    Routing information is stored as a dictionary of the form ``{"object_name":
    RouteMappingPair(method_mapping, routing_info)}``, where ``method_mapping``
    is an instance of :class:`~sklearn.utils.metadata_routing.MethodMapping` and
    ``routing_info`` is either a
    :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
    :class:`~sklearn.utils.metadata_routing.MetadataRouter` instance.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """
    _type = 'metadata_router'

    def __init__(self, owner):
        self._route_mappings = dict()
        self._self_request = None
        self.owner = owner

    def add_self_request(self, obj):
        """Add `self` (as a consumer) to the routing.

        This method is used if the router is also a consumer, and hence the
        router itself needs to be included in the routing. The passed object
        can be an estimator or a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest`.

        A router should add itself using this method instead of `add` since it
        should be treated differently than the other objects to which metadata
        is routed by the router.

        Parameters
        ----------
        obj : object
            This is typically the router instance, i.e. `self` in a
            ``get_metadata_routing()`` implementation. It can also be a
            ``MetadataRequest`` instance.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        if getattr(obj, '_type', None) == 'metadata_request':
            self._self_request = deepcopy(obj)
        elif hasattr(obj, '_get_metadata_request'):
            self._self_request = deepcopy(obj._get_metadata_request())
        else:
            raise ValueError('Given `obj` is neither a `MetadataRequest` nor does it implement the required API. Inheriting from `BaseEstimator` implements the required API.')
        return self

    def add(self, *, method_mapping, **objs):
        """Add named objects with their corresponding method mapping.

        Parameters
        ----------
        method_mapping : MethodMapping or str
            The mapping between the child and the parent's methods. If str, the
            output of :func:`~sklearn.utils.metadata_routing.MethodMapping.from_str`
            is used.

        **objs : dict
            A dictionary of objects from which metadata is extracted by calling
            :func:`~sklearn.utils.metadata_routing.get_routing_for_object` on them.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        if isinstance(method_mapping, str):
            method_mapping = MethodMapping.from_str(method_mapping)
        else:
            method_mapping = deepcopy(method_mapping)
        for name, obj in objs.items():
            self._route_mappings[name] = RouterMappingPair(mapping=method_mapping, router=get_routing_for_object(obj))
        return self

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
        res = set()
        if self._self_request:
            res = res | self._self_request.consumes(method=method, params=params)
        for _, route_mapping in self._route_mappings.items():
            for callee, caller in route_mapping.mapping:
                if caller == method:
                    res = res | route_mapping.router.consumes(method=callee, params=params)
        return res

    def _get_param_names(self, *, method, return_alias, ignore_self_request):
        """Get names of all metadata that can be consumed or routed by specified             method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned,
            which only applies to the stored `self`. If no `self` routing
            object is stored, this parameter has no effect.

        ignore_self_request : bool
            If `self._self_request` should be ignored. This is used in `_route_params`.
            If ``True``, ``return_alias`` has no effect.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        res = set()
        if self._self_request and (not ignore_self_request):
            res = res.union(self._self_request._get_param_names(method=method, return_alias=return_alias))
        for name, route_mapping in self._route_mappings.items():
            for callee, caller in route_mapping.mapping:
                if caller == method:
                    res = res.union(route_mapping.router._get_param_names(method=callee, return_alias=True, ignore_self_request=False))
        return res

    def _route_params(self, *, params, method):
        """Prepare the given parameters to be passed to the method.

        This is used when a router is used as a child object of another router.
        The parent router then passes all parameters understood by the child
        object to it and delegates their validation to the child.

        The output of this method can be used directly as the input to the
        corresponding method as extra props.

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
        res = Bunch()
        if self._self_request:
            res.update(self._self_request._route_params(params=params, method=method))
        param_names = self._get_param_names(method=method, return_alias=True, ignore_self_request=True)
        child_params = {key: value for key, value in params.items() if key in param_names}
        for key in set(res.keys()).intersection(child_params.keys()):
            if child_params[key] is not res[key]:
                raise ValueError(f'In {self.owner}, there is a conflict on {key} between what is requested for this estimator and what is requested by its children. You can resolve this conflict by using an alias for the child estimator(s) requested metadata.')
        res.update(child_params)
        return res

    def route_params(self, *, caller, params):
        """Return the input parameters requested by child objects.

        The output of this method is a bunch, which includes the inputs for all
        methods of each child object that are used in the router's `caller`
        method.

        If the router is also a consumer, it also checks for warnings of
        `self`'s/consumer's requested metadata.

        Parameters
        ----------
        caller : str
            The name of the method for which the parameters are requested and
            routed. If called inside the :term:`fit` method of a router, it
            would be `"fit"`.

        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of the form
            ``{"object_name": {"method_name": {prop: value}}}`` which can be
            used to pass the required metadata to corresponding methods or
            corresponding child objects.
        """
        if self._self_request:
            self._self_request._check_warnings(params=params, method=caller)
        res = Bunch()
        for name, route_mapping in self._route_mappings.items():
            router, mapping = (route_mapping.router, route_mapping.mapping)
            res[name] = Bunch()
            for _callee, _caller in mapping:
                if _caller == caller:
                    res[name][_callee] = router._route_params(params=params, method=_callee)
        return res

    def validate_metadata(self, *, method, params):
        """Validate given metadata for a method.

        This raises a ``TypeError`` if some of the passed metadata are not
        understood by child objects.

        Parameters
        ----------
        method : str
            The name of the method for which the parameters are requested and
            routed. If called inside the :term:`fit` method of a router, it
            would be `"fit"`.

        params : dict
            A dictionary of provided metadata.
        """
        param_names = self._get_param_names(method=method, return_alias=False, ignore_self_request=False)
        if self._self_request:
            self_params = self._self_request._get_param_names(method=method, return_alias=False)
        else:
            self_params = set()
        extra_keys = set(params.keys()) - param_names - self_params
        if extra_keys:
            raise TypeError(f'{self.owner}.{method} got unexpected argument(s) {extra_keys}, which are not requested metadata in any object.')

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        res = dict()
        if self._self_request:
            res['$self_request'] = self._self_request._serialize()
        for name, route_mapping in self._route_mappings.items():
            res[name] = dict()
            res[name]['mapping'] = route_mapping.mapping._serialize()
            res[name]['router'] = route_mapping.router._serialize()
        return res

    def __iter__(self):
        if self._self_request:
            yield ('$self_request', RouterMappingPair(mapping=MethodMapping.from_str('one-to-one'), router=self._self_request))
        for name, route_mapping in self._route_mappings.items():
            yield (name, route_mapping)

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))