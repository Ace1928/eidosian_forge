import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, convolve
from ..transform import integral_image
from .corner import structure_tensor
from ..morphology import octagon, star
from .censure_cy import _censure_dob_loop
from ..feature.util import (
from .._shared.utils import check_nD
class CENSURE(FeatureDetector):
    """CENSURE keypoint detector.

    min_scale : int, optional
        Minimum scale to extract keypoints from.
    max_scale : int, optional
        Maximum scale to extract keypoints from. The keypoints will be
        extracted from all the scales except the first and the last i.e.
        from the scales in the range [min_scale + 1, max_scale - 1]. The filter
        sizes for different scales is such that the two adjacent scales
        comprise of an octave.
    mode : {'DoB', 'Octagon', 'STAR'}, optional
        Type of bi-level filter used to get the scales of the input image.
        Possible values are 'DoB', 'Octagon' and 'STAR'. The three modes
        represent the shape of the bi-level filters i.e. box(square), octagon
        and star respectively. For instance, a bi-level octagon filter consists
        of a smaller inner octagon and a larger outer octagon with the filter
        weights being uniformly negative in both the inner octagon while
        uniformly positive in the difference region. Use STAR and Octagon for
        better features and DoB for better performance.
    non_max_threshold : float, optional
        Threshold value used to suppress maximas and minimas with a weak
        magnitude response obtained after Non-Maximal Suppression.
    line_threshold : float, optional
        Threshold for rejecting interest points which have ratio of principal
        curvatures greater than this value.

    Attributes
    ----------
    keypoints : (N, 2) array
        Keypoint coordinates as ``(row, col)``.
    scales : (N,) array
        Corresponding scales.

    References
    ----------
    .. [1] Motilal Agrawal, Kurt Konolige and Morten Rufus Blas
           "CENSURE: Center Surround Extremas for Realtime Feature
           Detection and Matching",
           https://link.springer.com/chapter/10.1007/978-3-540-88693-8_8
           :DOI:`10.1007/978-3-540-88693-8_8`

    .. [2] Adam Schmidt, Marek Kraft, Michal Fularz and Zuzanna Domagala
           "Comparative Assessment of Point Feature Detectors and
           Descriptors in the Context of Robot Navigation"
           http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-268aaf28-0faf-4872-a4df-7e2e61cb364c/c/Schmidt_comparative.pdf
           :DOI:`10.1.1.465.1117`

    Examples
    --------
    >>> from skimage.data import astronaut
    >>> from skimage.color import rgb2gray
    >>> from skimage.feature import CENSURE
    >>> img = rgb2gray(astronaut()[100:300, 100:300])
    >>> censure = CENSURE()
    >>> censure.detect(img)
    >>> censure.keypoints
    array([[  4, 148],
           [ 12,  73],
           [ 21, 176],
           [ 91,  22],
           [ 93,  56],
           [ 94,  22],
           [ 95,  54],
           [100,  51],
           [103,  51],
           [106,  67],
           [108,  15],
           [117,  20],
           [122,  60],
           [125,  37],
           [129,  37],
           [133,  76],
           [145,  44],
           [146,  94],
           [150, 114],
           [153,  33],
           [154, 156],
           [155, 151],
           [184,  63]])
    >>> censure.scales
    array([2, 6, 6, 2, 4, 3, 2, 3, 2, 6, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 4, 2,
           2])

    """

    def __init__(self, min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10):
        mode = mode.lower()
        if mode not in ('dob', 'octagon', 'star'):
            raise ValueError("`mode` must be one of 'DoB', 'Octagon', 'STAR'.")
        if min_scale < 1 or max_scale < 1 or max_scale - min_scale < 2:
            raise ValueError('The scales must be >= 1 and the number of scales should be >= 3.')
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode
        self.non_max_threshold = non_max_threshold
        self.line_threshold = line_threshold
        self.keypoints = None
        self.scales = None

    def detect(self, image):
        """Detect CENSURE keypoints along with the corresponding scale.

        Parameters
        ----------
        image : 2D ndarray
            Input image.

        """
        check_nD(image, 2)
        num_scales = self.max_scale - self.min_scale
        image = np.ascontiguousarray(_prepare_grayscale_input_2D(image))
        filter_response = _filter_image(image, self.min_scale, self.max_scale, self.mode)
        minimas = minimum_filter(filter_response, (3, 3, 3)) == filter_response
        maximas = maximum_filter(filter_response, (3, 3, 3)) == filter_response
        feature_mask = minimas | maximas
        feature_mask[filter_response < self.non_max_threshold] = False
        for i in range(1, num_scales):
            _suppress_lines(feature_mask[:, :, i], image, 1 + (self.min_scale + i - 1) / 3.0, self.line_threshold)
        rows, cols, scales = np.nonzero(feature_mask[..., 1:num_scales])
        keypoints = np.column_stack([rows, cols])
        scales = scales + self.min_scale + 1
        if self.mode == 'dob':
            self.keypoints = keypoints
            self.scales = scales
            return
        cumulative_mask = np.zeros(keypoints.shape[0], dtype=bool)
        if self.mode == 'octagon':
            for i in range(self.min_scale + 1, self.max_scale):
                c = (OCTAGON_OUTER_SHAPE[i - 1][0] - 1) // 2 + OCTAGON_OUTER_SHAPE[i - 1][1]
                cumulative_mask |= _mask_border_keypoints(image.shape, keypoints, c) & (scales == i)
        elif self.mode == 'star':
            for i in range(self.min_scale + 1, self.max_scale):
                c = STAR_SHAPE[STAR_FILTER_SHAPE[i - 1][0]] + STAR_SHAPE[STAR_FILTER_SHAPE[i - 1][0]] // 2
                cumulative_mask |= _mask_border_keypoints(image.shape, keypoints, c) & (scales == i)
        self.keypoints = keypoints[cumulative_mask]
        self.scales = scales[cumulative_mask]