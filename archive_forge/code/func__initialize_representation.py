import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def _initialize_representation(self, prefix=None):
    if prefix is None:
        prefix = self.prefix
    dtype = tools.prefix_dtype_map[prefix]
    if prefix not in self._representations:
        self._representations[prefix] = {}
        for matrix in self.shapes.keys():
            if matrix == 'obs':
                self._representations[prefix][matrix] = self.obs.astype(dtype)
            else:
                self._representations[prefix][matrix] = getattr(self, '_' + matrix).astype(dtype)
    else:
        for matrix in self.shapes.keys():
            existing = self._representations[prefix][matrix]
            if matrix == 'obs':
                pass
            else:
                new = getattr(self, '_' + matrix).astype(dtype)
                if existing.shape == new.shape:
                    existing[:] = new[:]
                else:
                    self._representations[prefix][matrix] = new
    if prefix in self._statespaces:
        ss = self._statespaces[prefix]
        create = not ss.obs.shape[1] == self.endog.shape[1] or not ss.design.shape[2] == self.design.shape[2] or (not ss.obs_intercept.shape[1] == self.obs_intercept.shape[1]) or (not ss.obs_cov.shape[2] == self.obs_cov.shape[2]) or (not ss.transition.shape[2] == self.transition.shape[2]) or (not ss.state_intercept.shape[1] == self.state_intercept.shape[1]) or (not ss.selection.shape[2] == self.selection.shape[2]) or (not ss.state_cov.shape[2] == self.state_cov.shape[2])
    else:
        create = True
    if create:
        if prefix in self._statespaces:
            del self._statespaces[prefix]
        cls = self.prefix_statespace_map[prefix]
        self._statespaces[prefix] = cls(self._representations[prefix]['obs'], self._representations[prefix]['design'], self._representations[prefix]['obs_intercept'], self._representations[prefix]['obs_cov'], self._representations[prefix]['transition'], self._representations[prefix]['state_intercept'], self._representations[prefix]['selection'], self._representations[prefix]['state_cov'])
    return (prefix, dtype, create)