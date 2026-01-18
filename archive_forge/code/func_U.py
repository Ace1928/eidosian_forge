import numpy as np
from pygsp import utils
@property
def U(self):
    """Fourier basis (eigenvectors of the Laplacian).

        Is computed by :func:`compute_fourier_basis`.
        """
    return self._check_fourier_properties('U', 'Fourier basis')