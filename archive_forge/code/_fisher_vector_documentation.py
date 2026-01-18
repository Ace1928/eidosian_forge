import numpy as np
Compute the Fisher vector given some descriptors/vectors,
    and an associated estimated GMM.

    Parameters
    ----------
    descriptors : np.ndarray, shape=(n_descriptors, descriptor_length)
        NumPy array of the descriptors for which the Fisher vector
        representation is to be computed.
    gmm : :class:`sklearn.mixture.GaussianMixture`
        An estimated GMM object, which contains the necessary parameters needed
        to compute the Fisher vector.
    improved : bool, default=False
        Flag denoting whether to compute improved Fisher vectors or not.
        Improved Fisher vectors are L2 and power normalized. Power
        normalization is simply f(z) = sign(z) pow(abs(z), alpha) for some
        0 <= alpha <= 1.
    alpha : float, default=0.5
        The parameter for the power normalization step. Ignored if
        improved=False.

    Returns
    -------
    fisher_vector : np.ndarray
        The computation Fisher vector, which is given by a concatenation of the
        gradients of a GMM with respect to its parameters (mixture weights,
        means, and covariance matrices). For D-dimensional input descriptors or
        vectors, and a K-mode GMM, the Fisher vector dimensionality will be
        2KD + K. Thus, its dimensionality is invariant to the number of
        descriptors/vectors.

    References
    ----------
    .. [1] Perronnin, F. and Dance, C. Fisher kernels on Visual Vocabularies
           for Image Categorization, IEEE Conference on Computer Vision and
           Pattern Recognition, 2007
    .. [2] Perronnin, F. and Sanchez, J. and Mensink T. Improving the Fisher
           Kernel for Large-Scale Image Classification, ECCV, 2010

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('sklearn')

    >>> from skimage.feature import fisher_vector, learn_gmm
    >>> sift_for_images = [np.random.random((10, 128)) for _ in range(10)]
    >>> num_modes = 16
    >>> # Estimate 16-mode GMM with these synthetic SIFT vectors
    >>> gmm = learn_gmm(sift_for_images, n_modes=num_modes)
    >>> test_image_descriptors = np.random.random((25, 128))
    >>> # Compute the Fisher vector
    >>> fv = fisher_vector(test_image_descriptors, gmm)
    