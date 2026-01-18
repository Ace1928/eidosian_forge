import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
class _GeometricTransform(ABC):
    """Abstract base class for geometric transformations."""

    @abstractmethod
    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.

        """

    @property
    @abstractmethod
    def inverse(self):
        """Return a transform object representing the inverse."""

    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the Euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N,) array
            Residual for coordinate.

        """
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))