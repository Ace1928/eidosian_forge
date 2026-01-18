import numpy as np
from pygsp import utils
def _check_fourier_properties(self, name, desc):
    if not hasattr(self, '_' + name):
        self.logger.warning('The {} G.{} is not available, we need to compute the Fourier basis. Explicitly call G.compute_fourier_basis() once beforehand to suppress the warning.'.format(desc, name))
        self.compute_fourier_basis()
    return getattr(self, '_' + name)