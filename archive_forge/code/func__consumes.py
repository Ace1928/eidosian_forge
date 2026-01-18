import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def _consumes(self, params):
    """Check whether the given parameters are consumed by this method.

        Parameters
        ----------
        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by this method.
        """
    params = set(params)
    res = set()
    for prop, alias in self._requests.items():
        if alias is True and prop in params:
            res.add(prop)
        elif isinstance(alias, str) and alias in params:
            res.add(alias)
    return res