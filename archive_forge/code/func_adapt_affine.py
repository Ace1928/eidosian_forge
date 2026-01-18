import numpy as np
import numpy.linalg as npl
from .optpkg import optional_package
from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox
def adapt_affine(affine, n_dim):
    """Adapt input / output dimensions of spatial `affine` for `n_dims`

    Adapts a spatial (4, 4) affine that is being applied to an image with fewer
    than 3 spatial dimensions, or more than 3 dimensions.  If there are more
    than three dimensions, assume an identity transformation for these
    dimensions.

    Parameters
    ----------
    affine : array-like
        affine transform. Usually shape (4, 4).  For what follows ``N, M =
        affine.shape``
    n_dims : int
        Number of dimensions of underlying array, and therefore number of input
        dimensions for affine.

    Returns
    -------
    adapted : shape (M, n_dims+1) array
        Affine array adapted to number of input dimensions.  Columns of the
        affine corresponding to missing input dimensions have been dropped,
        columns corresponding to extra input dimensions have an extra identity
        column added
    """
    affine = np.asarray(affine)
    rzs, trans = to_matvec(affine)
    rzs = rzs[:, :n_dim]
    adapted = from_matvec(rzs, trans)
    n_extra_columns = n_dim - adapted.shape[1] + 1
    if n_extra_columns > 0:
        adapted = append_diag(adapted, np.ones((n_extra_columns,)))
    return adapted