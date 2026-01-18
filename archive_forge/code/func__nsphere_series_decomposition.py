import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _nsphere_series_decomposition(radius, ndim, dtype=np.uint8):
    """Generate a sequence of footprints approximating an n-sphere.

    Morphological operations with an n-sphere (hypersphere) footprint can be
    approximated by applying a series of smaller footprints of extent 3 along
    each axis. Specific solutions for this are given in [1]_ for the case of
    2D disks with radius 2 through 10.

    Here we used n-dimensional extensions of the "square", "diamond" and
    "t-shaped" elements from that publication. All of these elementary elements
    have size ``(3,) * ndim``. We numerically computed the number of
    repetitions of each element that gives the closest match to the disk
    (in 2D) or ball (in 3D) computed with ``decomposition=None``.

    The approach can be extended to higher dimensions, but we have only stored
    results for 2D and 3D at this point.

    Empirically, the shapes at large radius approach a hexadecagon
    (16-sides [2]_) in 2D and a rhombicuboctahedron (26-faces, [3]_) in 3D.

    References
    ----------
    .. [1] Park, H and Chin R.T. Decomposition of structuring elements for
           optimal implementation of morphological operations. In Proceedings:
           1997 IEEE Workshop on Nonlinear Signal and Image Processing, London,
           UK.
           https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970226.pdf
    .. [2] https://en.wikipedia.org/wiki/Hexadecagon
    .. [3] https://en.wikipedia.org/wiki/Rhombicuboctahedron
    """
    if radius == 1:
        kwargs = dict(dtype=dtype, strict_radius=False, decomposition=None)
        if ndim == 2:
            return ((disk(1, **kwargs), 1),)
        elif ndim == 3:
            return ((ball(1, **kwargs), 1),)
    if ndim not in _nsphere_decompositions:
        raise ValueError('sequence decompositions are only currently available for 2d disks or 3d balls')
    precomputed_decompositions = _nsphere_decompositions[ndim]
    max_radius = precomputed_decompositions.shape[0]
    if radius > max_radius:
        raise ValueError(f'precomputed {ndim}D decomposition unavailable for radius > {max_radius}')
    num_t_series, num_diamond, num_square = precomputed_decompositions[radius]
    sequence = []
    if num_t_series > 0:
        all_t = _t_shaped_element_series(ndim=ndim, dtype=dtype)
        [sequence.append((t, num_t_series)) for t in all_t]
    if num_diamond > 0:
        d = np.zeros((3,) * ndim, dtype=dtype)
        sl = [slice(1, 2)] * ndim
        for ax in range(ndim):
            sl[ax] = slice(None)
            d[tuple(sl)] = 1
            sl[ax] = slice(1, 2)
        sequence.append((d, num_diamond))
    if num_square > 0:
        sq = np.ones((3,) * ndim, dtype=dtype)
        sequence.append((sq, num_square))
    return tuple(sequence)