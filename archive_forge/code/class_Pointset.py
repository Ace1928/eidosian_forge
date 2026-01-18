from __future__ import annotations
import math
import typing as ty
from dataclasses import dataclass, replace
import numpy as np
from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage
@dataclass
class Pointset:
    """A collection of points described by coordinates.

    Parameters
    ----------
    coords : array-like
      (*N*, *n*) array with *N* being points and columns their *n*-dimensional coordinates
    affine : :class:`numpy.ndarray`
      Affine transform to be applied to coordinates array
    homogeneous : :class:`bool`
      Indicate whether the provided coordinates are homogeneous,
      i.e., homogeneous 3D coordinates have the form ``(x, y, z, 1)``
    """
    coordinates: CoordinateArray
    affine: np.ndarray
    homogeneous: bool = False
    __array_priority__ = 99

    def __init__(self, coordinates: CoordinateArray, affine: np.ndarray | None=None, homogeneous: bool=False):
        self.coordinates = coordinates
        self.homogeneous = homogeneous
        if affine is None:
            self.affine = np.eye(self.dim + 1)
        else:
            self.affine = np.asanyarray(affine)
        if self.affine.shape != (self.dim + 1,) * 2:
            raise ValueError(f'Invalid affine for {self.dim}D coordinates:\n{self.affine}')
        if np.any(self.affine[-1, :-1] != 0) or self.affine[-1, -1] != 1:
            raise ValueError(f'Invalid affine matrix:\n{self.affine}')

    @property
    def n_coords(self) -> int:
        """Number of coordinates

        Subclasses should override with more efficient implementations.
        """
        return self.coordinates.shape[0]

    @property
    def dim(self) -> int:
        """The dimensionality of the space the coordinates are in"""
        return self.coordinates.shape[1] - self.homogeneous

    def __rmatmul__(self, affine: np.ndarray) -> Self:
        """Apply an affine transformation to the pointset

        This will return a new pointset with an updated affine matrix only.
        """
        return replace(self, affine=np.asanyarray(affine) @ self.affine)

    def _homogeneous_coords(self):
        if self.homogeneous:
            return np.asanyarray(self.coordinates)
        ones = strided_scalar(shape=(self.coordinates.shape[0], 1), scalar=np.array(1, dtype=self.coordinates.dtype))
        return np.hstack((self.coordinates, ones))

    def get_coords(self, *, as_homogeneous: bool=False):
        """Retrieve the coordinates

        Parameters
        ----------
        as_homogeneous : :class:`bool`
            Return homogeneous coordinates if ``True``, or Cartesian
            coordinates if ``False``.

        name : :class:`str`
            Select a particular coordinate system if more than one may exist.
            By default, `None` is equivalent to `"world"` and corresponds to
            an RAS+ coordinate system.
        """
        ident = np.allclose(self.affine, np.eye(self.affine.shape[0]))
        if self.homogeneous == as_homogeneous and ident:
            return np.asanyarray(self.coordinates)
        coords = self._homogeneous_coords()
        if not ident:
            coords = (self.affine @ coords.T).T
        if not as_homogeneous:
            coords = coords[:, :-1]
        return coords