import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
Chan-Vese segmentation algorithm.

    Active contour model by evolving a level set. Can be used to
    segment objects without clearly defined boundaries.

    Parameters
    ----------
    image : (M, N) ndarray
        Grayscale image to be segmented.
    mu : float, optional
        'edge length' weight parameter. Higher `mu` values will
        produce a 'round' edge, while values closer to zero will
        detect smaller objects.
    lambda1 : float, optional
        'difference from average' weight parameter for the output
        region with value 'True'. If it is lower than `lambda2`, this
        region will have a larger range of values than the other.
    lambda2 : float, optional
        'difference from average' weight parameter for the output
        region with value 'False'. If it is lower than `lambda1`, this
        region will have a larger range of values than the other.
    tol : float, positive, optional
        Level set variation tolerance between iterations. If the
        L2 norm difference between the level sets of successive
        iterations normalized by the area of the image is below this
        value, the algorithm will assume that the solution was
        reached.
    max_num_iter : uint, optional
        Maximum number of iterations allowed before the algorithm
        interrupts itself.
    dt : float, optional
        A multiplication factor applied at calculations for each step,
        serves to accelerate the algorithm. While higher values may
        speed up the algorithm, they may also lead to convergence
        problems.
    init_level_set : str or (M, N) ndarray, optional
        Defines the starting level set used by the algorithm.
        If a string is inputted, a level set that matches the image
        size will automatically be generated. Alternatively, it is
        possible to define a custom level set, which should be an
        array of float values, with the same shape as 'image'.
        Accepted string values are as follows.

        'checkerboard'
            the starting level set is defined as
            sin(x/5*pi)*sin(y/5*pi), where x and y are pixel
            coordinates. This level set has fast convergence, but may
            fail to detect implicit edges.
        'disk'
            the starting level set is defined as the opposite
            of the distance from the center of the image minus half of
            the minimum value between image width and image height.
            This is somewhat slower, but is more likely to properly
            detect implicit edges.
        'small disk'
            the starting level set is defined as the
            opposite of the distance from the center of the image
            minus a quarter of the minimum value between image width
            and image height.
    extended_output : bool, optional
        If set to True, the return value will be a tuple containing
        the three return values (see below). If set to False which
        is the default value, only the 'segmentation' array will be
        returned.

    Returns
    -------
    segmentation : (M, N) ndarray, bool
        Segmentation produced by the algorithm.
    phi : (M, N) ndarray of floats
        Final level set computed by the algorithm.
    energies : list of floats
        Shows the evolution of the 'energy' for each step of the
        algorithm. This should allow to check whether the algorithm
        converged.

    Notes
    -----
    The Chan-Vese Algorithm is designed to segment objects without
    clearly defined boundaries. This algorithm is based on level sets
    that are evolved iteratively to minimize an energy, which is
    defined by weighted values corresponding to the sum of differences
    intensity from the average value outside the segmented region, the
    sum of differences from the average value inside the segmented
    region, and a term which is dependent on the length of the
    boundary of the segmented region.

    This algorithm was first proposed by Tony Chan and Luminita Vese,
    in a publication entitled "An Active Contour Model Without Edges"
    [1]_.

    This implementation of the algorithm is somewhat simplified in the
    sense that the area factor 'nu' described in the original paper is
    not implemented, and is only suitable for grayscale images.

    Typical values for `lambda1` and `lambda2` are 1. If the
    'background' is very different from the segmented object in terms
    of distribution (for example, a uniform black image with figures
    of varying intensity), then these values should be different from
    each other.

    Typical values for mu are between 0 and 1, though higher values
    can be used when dealing with shapes with very ill-defined
    contours.

    The 'energy' which this algorithm tries to minimize is defined
    as the sum of the differences from the average within the region
    squared and weighed by the 'lambda' factors to which is added the
    length of the contour multiplied by the 'mu' factor.

    Supports 2D grayscale images only, and does not implement the area
    term described in the original article.

    References
    ----------
    .. [1] An Active Contour Model without Edges, Tony Chan and
           Luminita Vese, Scale-Space Theories in Computer Vision,
           1999, :DOI:`10.1007/3-540-48236-9_13`
    .. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On
           Line, 2 (2012), pp. 214-224,
           :DOI:`10.5201/ipol.2012.g-cv`
    .. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen, 2011
           :arXiv:`1107.2782`
    