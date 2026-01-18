import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def _raise_for_unsupported_routing(obj, method, **kwargs):
    """Raise when metadata routing is enabled and metadata is passed.

    This is used in meta-estimators which have not implemented metadata routing
    to prevent silent bugs. There is no need to use this function if the
    meta-estimator is not accepting any metadata, especially in `fit`, since
    if a meta-estimator accepts any metadata, they would do that in `fit` as
    well.

    Parameters
    ----------
    obj : estimator
        The estimator for which we're raising the error.

    method : str
        The method where the error is raised.

    **kwargs : dict
        The metadata passed to the method.
    """
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if _routing_enabled() and kwargs:
        cls_name = obj.__class__.__name__
        raise NotImplementedError(f'{cls_name}.{method} cannot accept given metadata ({set(kwargs.keys())}) since metadata routing is not yet implemented for {cls_name}.')