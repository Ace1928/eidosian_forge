import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def _raise_for_params(params, owner, method):
    """Raise an error if metadata routing is not enabled and params are passed.

    .. versionadded:: 1.4

    Parameters
    ----------
    params : dict
        The metadata passed to a method.

    owner : object
        The object to which the method belongs.

    method : str
        The name of the method, e.g. "fit".

    Raises
    ------
    ValueError
        If metadata routing is not enabled and params are passed.
    """
    caller = f'{owner.__class__.__name__}.{method}' if method else owner.__class__.__name__
    if not _routing_enabled() and params:
        raise ValueError(f'Passing extra keyword arguments to {caller} is only supported if enable_metadata_routing=True, which you can set using `sklearn.set_config`. See the User Guide <https://scikit-learn.org/stable/metadata_routing.html> for more details. Extra parameters passed are: {set(params)}')