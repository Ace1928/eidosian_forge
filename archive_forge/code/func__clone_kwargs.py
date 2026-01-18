import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def _clone_kwargs(self, endog, **kwargs):
    """
        Construct keyword arguments for cloning a state space model

        Parameters
        ----------
        endog : array_like
            An observed time-series process :math:`y`.
        **kwargs
            Keyword arguments to pass to the new state space representation
            model constructor. Those that are not specified are copied from
            the specification of the current state space model.
        """
    kwargs['nobs'] = len(endog)
    kwargs['k_endog'] = self.k_endog
    for key in ['k_states', 'k_posdef']:
        val = getattr(self, key)
        if key not in kwargs or kwargs[key] is None:
            kwargs[key] = val
        if kwargs[key] != val:
            raise ValueError('Cannot change the dimension of %s when cloning.' % key)
    for name in self.shapes.keys():
        if name == 'obs':
            continue
        if name not in kwargs:
            mat = getattr(self, name)
            if mat.shape[-1] != 1:
                raise ValueError('The `%s` matrix is time-varying. Cloning this model requires specifying an updated matrix.' % name)
            kwargs[name] = mat
    kwargs.setdefault('initialization', self.initialization)
    return kwargs