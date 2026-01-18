import warnings
import numpy as np
from . import tools
def _initialize_initialization(self, prefix):
    dtype = tools.prefix_dtype_map[prefix]
    if prefix not in self._representations:
        self._representations[prefix] = {'constant': self.constant.astype(dtype), 'stationary_cov': np.asfortranarray(self.stationary_cov.astype(dtype))}
    else:
        self._representations[prefix]['constant'][:] = self.constant.astype(dtype)[:]
        self._representations[prefix]['stationary_cov'][:] = self.stationary_cov.astype(dtype)[:]
    if prefix not in self._initializations:
        cls = self.prefix_initialization_map[prefix]
        self._initializations[prefix] = cls(self.k_states, self._representations[prefix]['constant'], self._representations[prefix]['stationary_cov'], self.approximate_diffuse_variance)
    else:
        self._initializations[prefix].approximate_diffuse_variance = self.approximate_diffuse_variance
    return (prefix, dtype)