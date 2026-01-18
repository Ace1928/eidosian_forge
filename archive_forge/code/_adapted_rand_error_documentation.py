from .._shared.utils import check_shape_equality
from ._contingency_table import contingency_table
Compute Adapted Rand error as defined by the SNEMI3D contest. [1]_

    Parameters
    ----------
    image_true : ndarray of int
        Ground-truth label image, same shape as im_test.
    image_test : ndarray of int
        Test image.
    table : scipy.sparse array in crs format, optional
        A contingency table built with skimage.evaluate.contingency_table.
        If None, it will be computed on the fly.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.
    alpha : float, optional
        Relative weight given to precision and recall in the adapted Rand error
        calculation.

    Returns
    -------
    are : float
        The adapted Rand error.
    prec : float
        The adapted Rand precision: this is the number of pairs of pixels that
        have the same label in the test label image *and* in the true image,
        divided by the number in the test image.
    rec : float
        The adapted Rand recall: this is the number of pairs of pixels that
        have the same label in the test label image *and* in the true image,
        divided by the number in the true image.

    Notes
    -----
    Pixels with label 0 in the true segmentation are ignored in the score.

    The adapted Rand error is calculated as follows:

    :math:`1 - \frac{\sum_{ij} p_{ij}^{2}}{\alpha \sum_{k} s_{k}^{2} +
    (1-\alpha)\sum_{k} t_{k}^{2}}`,
    where :math:`p_{ij}` is the probability that a pixel has the same label
    in the test image *and* in the true image, :math:`t_{k}` is the
    probability that a pixel has label :math:`k` in the true image,
    and :math:`s_{k}` is the probability that a pixel has label :math:`k`
    in the test image.

    Default behavior is to weight precision and recall equally in the
    adapted Rand error calculation.
    When alpha = 0, adapted Rand error = recall.
    When alpha = 1, adapted Rand error = precision.


    References
    ----------
    .. [1] Arganda-Carreras I, Turaga SC, Berger DR, et al. (2015)
           Crowdsourcing the creation of image segmentation algorithms
           for connectomics. Front. Neuroanat. 9:142.
           :DOI:`10.3389/fnana.2015.00142`
    