import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
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