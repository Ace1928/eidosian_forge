import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
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