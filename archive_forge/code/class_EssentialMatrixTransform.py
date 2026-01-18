import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
class EssentialMatrixTransform(FundamentalMatrixTransform):
    """Essential matrix transformation.

    The essential matrix relates corresponding points between a pair of
    calibrated images. The matrix transforms normalized, homogeneous image
    points in one image to epipolar lines in the other image.

    The essential matrix is only defined for a pair of moving images capturing a
    non-planar scene. In the case of pure rotation or planar scenes, the
    homography describes the geometric relation between two images
    (`ProjectiveTransform`). If the intrinsic calibration of the images is
    unknown, the fundamental matrix describes the projective relation between
    the two images (`FundamentalMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    rotation : (3, 3) array_like, optional
        Rotation matrix of the relative camera motion.
    translation : (3, 1) array_like, optional
        Translation vector of the relative camera motion. The vector must
        have unit length.
    matrix : (3, 3) array_like, optional
        Essential matrix.

    Attributes
    ----------
    params : (3, 3) array
        Essential matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>>
    >>> tform_matrix = ski.transform.EssentialMatrixTransform(
    ...     rotation=np.eye(3), translation=np.array([0, 0, 1])
    ... )
    >>> tform_matrix.params
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> src = np.array([[ 1.839035, 1.924743],
    ...                 [ 0.543582, 0.375221],
    ...                 [ 0.47324 , 0.142522],
    ...                 [ 0.96491 , 0.598376],
    ...                 [ 0.102388, 0.140092],
    ...                 [15.994343, 9.622164],
    ...                 [ 0.285901, 0.430055],
    ...                 [ 0.09115 , 0.254594]])
    >>> dst = np.array([[1.002114, 1.129644],
    ...                 [1.521742, 1.846002],
    ...                 [1.084332, 0.275134],
    ...                 [0.293328, 0.588992],
    ...                 [0.839509, 0.08729 ],
    ...                 [1.779735, 1.116857],
    ...                 [0.878616, 0.602447],
    ...                 [0.642616, 1.028681]])
    >>> tform_matrix.estimate(src, dst)
    True
    >>> tform_matrix.residuals(src, dst)
    array([0.42455187, 0.01460448, 0.13847034, 0.12140951, 0.27759346,
           0.32453118, 0.00210776, 0.26512283])

    """

    def __init__(self, rotation=None, translation=None, matrix=None, *, dimensionality=2):
        super().__init__(matrix=matrix, dimensionality=dimensionality)
        if rotation is not None:
            rotation = np.asarray(rotation)
            if translation is None:
                raise ValueError('Both rotation and translation required')
            translation = np.asarray(translation)
            if rotation.shape != (3, 3):
                raise ValueError('Invalid shape of rotation matrix')
            if abs(np.linalg.det(rotation) - 1) > 1e-06:
                raise ValueError('Rotation matrix must have unit determinant')
            if translation.size != 3:
                raise ValueError('Invalid shape of translation vector')
            if abs(np.linalg.norm(translation) - 1) > 1e-06:
                raise ValueError('Translation vector must have unit length')
            t_x = np.array([0, -translation[2], translation[1], translation[2], 0, -translation[0], -translation[1], translation[0], 0]).reshape(3, 3)
            self.params = t_x @ rotation
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape != (3, 3):
                raise ValueError('Invalid shape of transformation matrix')
            self.params = matrix
        else:
            self.params = np.eye(3)

    def estimate(self, src, dst):
        """Estimate essential matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        E_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(src, dst)
        U, S, V = np.linalg.svd(E_normalized)
        S[0] = (S[0] + S[1]) / 2.0
        S[1] = S[0]
        S[2] = 0
        E = U @ np.diag(S) @ V
        self.params = dst_matrix.T @ E @ src_matrix
        return True