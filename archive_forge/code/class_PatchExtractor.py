from itertools import product
from numbers import Integral, Number, Real
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Hidden, Interval, RealNotInt, validate_params
class PatchExtractor(TransformerMixin, BaseEstimator):
    """Extracts patches from a collection of images.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    patch_size : tuple of int (patch_height, patch_width), default=None
        The dimensions of one patch. If set to None, the patch size will be
        automatically set to `(img_height // 10, img_width // 10)`, where
        `img_height` and `img_width` are the dimensions of the input images.

    max_patches : int or float, default=None
        The maximum number of patches per image to extract. If `max_patches` is
        a float in (0, 1), it is taken to mean a proportion of the total number
        of patches. If set to None, extract all possible patches.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches is not None`. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    See Also
    --------
    reconstruct_from_patches_2d : Reconstruct image from all of its patches.

    Notes
    -----
    This estimator is stateless and does not need to be fitted. However, we
    recommend to call :meth:`fit_transform` instead of :meth:`transform`, as
    parameter validation is only performed in :meth:`fit`.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_images
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the second image in this dataset:
    >>> X = load_sample_images().images[1]
    >>> X = X[None, ...]
    >>> print(f"Image shape: {X.shape}")
    Image shape: (1, 427, 640, 3)
    >>> pe = image.PatchExtractor(patch_size=(10, 10))
    >>> pe_trans = pe.transform(X)
    >>> print(f"Patches shape: {pe_trans.shape}")
    Patches shape: (263758, 10, 10, 3)
    >>> X_reconstructed = image.reconstruct_from_patches_2d(pe_trans, X.shape[1:])
    >>> print(f"Reconstructed shape: {X_reconstructed.shape}")
    Reconstructed shape: (427, 640, 3)
    """
    _parameter_constraints: dict = {'patch_size': [tuple, None], 'max_patches': [None, Interval(RealNotInt, 0, 1, closed='neither'), Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state']}

    def __init__(self, *, patch_size=None, max_patches=None, random_state=None):
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Only validate the parameters of the estimator.

        This method allows to: (i) validate the parameters of the estimator  and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : ndarray of shape (n_samples, image_height, image_width) or                 (n_samples, image_height, image_width, n_channels)
            Array of images from which to extract patches. For color images,
            the last dimension specifies the channel: a RGB image would have
            `n_channels=3`.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """Transform the image samples in `X` into a matrix of patch data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, image_height, image_width) or                 (n_samples, image_height, image_width, n_channels)
            Array of images from which to extract patches. For color images,
            the last dimension specifies the channel: a RGB image would have
            `n_channels=3`.

        Returns
        -------
        patches : array of shape (n_patches, patch_height, patch_width) or                 (n_patches, patch_height, patch_width, n_channels)
            The collection of patches extracted from the images, where
            `n_patches` is either `n_samples * max_patches` or the total
            number of patches that can be extracted.
        """
        X = self._validate_data(X=X, ensure_2d=False, allow_nd=True, ensure_min_samples=1, ensure_min_features=1, reset=False)
        random_state = check_random_state(self.random_state)
        n_imgs, img_height, img_width = X.shape[:3]
        if self.patch_size is None:
            patch_size = (img_height // 10, img_width // 10)
        else:
            if len(self.patch_size) != 2:
                raise ValueError(f'patch_size must be a tuple of two integers. Got {self.patch_size} instead.')
            patch_size = self.patch_size
        n_imgs, img_height, img_width = X.shape[:3]
        X = np.reshape(X, (n_imgs, img_height, img_width, -1))
        n_channels = X.shape[-1]
        patch_height, patch_width = patch_size
        n_patches = _compute_n_patches(img_height, img_width, patch_height, patch_width, self.max_patches)
        patches_shape = (n_imgs * n_patches,) + patch_size
        if n_channels > 1:
            patches_shape += (n_channels,)
        patches = np.empty(patches_shape)
        for ii, image in enumerate(X):
            patches[ii * n_patches:(ii + 1) * n_patches] = extract_patches_2d(image, patch_size, max_patches=self.max_patches, random_state=random_state)
        return patches

    def _more_tags(self):
        return {'X_types': ['3darray'], 'stateless': True}