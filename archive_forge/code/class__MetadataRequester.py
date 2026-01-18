import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class _MetadataRequester:
    """Mixin class for adding metadata request functionality.

    ``BaseEstimator`` inherits from this Mixin.

    .. versionadded:: 1.3
    """
    if TYPE_CHECKING:

        def set_fit_request(self, **kwargs):
            pass

        def set_partial_fit_request(self, **kwargs):
            pass

        def set_predict_request(self, **kwargs):
            pass

        def set_predict_proba_request(self, **kwargs):
            pass

        def set_predict_log_proba_request(self, **kwargs):
            pass

        def set_decision_function_request(self, **kwargs):
            pass

        def set_score_request(self, **kwargs):
            pass

        def set_split_request(self, **kwargs):
            pass

        def set_transform_request(self, **kwargs):
            pass

        def set_inverse_transform_request(self, **kwargs):
            pass

    def __init_subclass__(cls, **kwargs):
        """Set the ``set_{method}_request`` methods.

        This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It
        looks for the information available in the set default values which are
        set using ``__metadata_request__*`` class attributes, or inferred
        from method signatures.

        The ``__metadata_request__*`` class attributes are used when a method
        does not explicitly accept a metadata through its arguments or if the
        developer would like to specify a request value for those metadata
        which are different from the default ``None``.

        References
        ----------
        .. [1] https://www.python.org/dev/peps/pep-0487
        """
        try:
            requests = cls._get_default_requests()
        except Exception:
            super().__init_subclass__(**kwargs)
            return
        for method in SIMPLE_METHODS:
            mmr = getattr(requests, method)
            if not len(mmr.requests):
                continue
            setattr(cls, f'set_{method}_request', RequestMethod(method, sorted(mmr.requests.keys())))
        super().__init_subclass__(**kwargs)

    @classmethod
    def _build_request_for_signature(cls, router, method):
        """Build the `MethodMetadataRequest` for a method using its signature.

        This method takes all arguments from the method signature and uses
        ``None`` as their default request value, except ``X``, ``y``, ``Y``,
        ``Xt``, ``yt``, ``*args``, and ``**kwargs``.

        Parameters
        ----------
        router : MetadataRequest
            The parent object for the created `MethodMetadataRequest`.
        method : str
            The name of the method.

        Returns
        -------
        method_request : MethodMetadataRequest
            The prepared request using the method's signature.
        """
        mmr = MethodMetadataRequest(owner=cls.__name__, method=method)
        if not hasattr(cls, method) or not inspect.isfunction(getattr(cls, method)):
            return mmr
        params = list(inspect.signature(getattr(cls, method)).parameters.items())[1:]
        for pname, param in params:
            if pname in {'X', 'y', 'Y', 'Xt', 'yt'}:
                continue
            if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
                continue
            mmr.add_request(param=pname, alias=None)
        return mmr

    @classmethod
    def _get_default_requests(cls):
        """Collect default request values.

        This method combines the information present in ``__metadata_request__*``
        class attributes, as well as determining request keys from method
        signatures.
        """
        requests = MetadataRequest(owner=cls.__name__)
        for method in SIMPLE_METHODS:
            setattr(requests, method, cls._build_request_for_signature(router=requests, method=method))
        defaults = dict()
        for base_class in reversed(inspect.getmro(cls)):
            base_defaults = {attr: value for attr, value in vars(base_class).items() if '__metadata_request__' in attr}
            defaults.update(base_defaults)
        defaults = dict(sorted(defaults.items()))
        for attr, value in defaults.items():
            substr = '__metadata_request__'
            method = attr[attr.index(substr) + len(substr):]
            for prop, alias in value.items():
                getattr(requests, method).add_request(param=prop, alias=alias)
        return requests

    def _get_metadata_request(self):
        """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        request : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.
        """
        if hasattr(self, '_metadata_request'):
            requests = get_routing_for_object(self._metadata_request)
        else:
            requests = self._get_default_requests()
        return requests

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
            routing information.
        """
        return self._get_metadata_request()