import base64
import numpy as np
from . import _marching_cubes_lewiner_luts as mcluts
from . import _marching_cubes_lewiner_cy
Compute surface area, given vertices and triangular faces.

    Parameters
    ----------
    verts : (V, 3) array of floats
        Array containing coordinates for V unique mesh vertices.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in `verts`.

    Returns
    -------
    area : float
        Surface area of mesh. Units now [coordinate units] ** 2.

    Notes
    -----
    The arguments expected by this function are the first two outputs from
    `skimage.measure.marching_cubes`. For unit correct output, ensure correct
    `spacing` was passed to `skimage.measure.marching_cubes`.

    This algorithm works properly only if the ``faces`` provided are all
    triangles.

    See Also
    --------
    skimage.measure.marching_cubes

    