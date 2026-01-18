import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def CreateDetAugmenter(data_shape, resize=0, rand_crop=0, rand_pad=0, rand_gray=0, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0, saturation=0, pca_noise=0, hue=0, inter_method=2, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 3.0), min_eject_coverage=0.3, max_attempts=50, pad_val=(127, 127, 127)):
    """Create augmenters for detection.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    resize : int
        Resize shorter edge if larger than 0 at the begining
    rand_crop : float
        [0, 1], probability to apply random cropping
    rand_pad : float
        [0, 1], probability to apply random padding
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels
    rand_mirror : bool
        Whether to apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        Mean pixel values for [r, g, b]
    std : np.ndarray or None
        Standard deviations for [r, g, b]
    brightness : float
        Brightness jittering range (percent)
    contrast : float
        Contrast jittering range (percent)
    saturation : float
        Saturation jittering range (percent)
    hue : float
        Hue jittering range (percent)
    pca_noise : float
        Pca noise level (percent)
    inter_method : int, default=2(Area-based)
        Interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
    min_object_covered : float
        The cropped area of the image must contain at least this fraction of
        any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the
        bounding boxes supplied.
    min_eject_coverage : float
        The minimum coverage of cropped sample w.r.t its original size. With this
        constraint, objects that have marginal area after crop will be discarded.
    aspect_ratio_range : tuple of floats
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    pad_val: float
        Pixel value to be filled when padding is enabled. pad_val will automatically
        be subtracted by mean and divided by std if applicable.

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.image.CreateDetAugmenter(data_shape=(3, 300, 300), rand_crop=0.5,
    ...    rand_pad=0.5, rand_mirror=True, mean=True, brightness=0.125, contrast=0.125,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10, min_object_covered=[0.3, 0.5, 0.9],
    ...    area_range=(0.3, 3.0))
    >>> # dump the details
    >>> for aug in augs:
    ...    aug.dumps()
    """
    auglist = []
    if resize > 0:
        auglist.append(DetBorrowAug(ResizeAug(resize, inter_method)))
    if rand_crop > 0:
        crop_augs = CreateMultiRandCropAugmenter(min_object_covered, aspect_ratio_range, area_range, min_eject_coverage, max_attempts, skip_prob=1 - rand_crop)
        auglist.append(crop_augs)
    if rand_mirror > 0:
        auglist.append(DetHorizontalFlipAug(0.5))
    if rand_pad > 0:
        pad_aug = DetRandomPadAug(aspect_ratio_range, (1.0, area_range[1]), max_attempts, pad_val)
        auglist.append(DetRandomSelectAug([pad_aug], 1 - rand_pad))
    auglist.append(DetBorrowAug(ForceResizeAug((data_shape[2], data_shape[1]), inter_method)))
    auglist.append(DetBorrowAug(CastAug()))
    if brightness or contrast or saturation:
        auglist.append(DetBorrowAug(ColorJitterAug(brightness, contrast, saturation)))
    if hue:
        auglist.append(DetBorrowAug(HueJitterAug(hue)))
    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.814], [-0.5836, -0.6948, 0.4203]])
        auglist.append(DetBorrowAug(LightingAug(pca_noise, eigval, eigvec)))
    if rand_gray > 0:
        auglist.append(DetBorrowAug(RandomGrayAug(rand_gray)))
    if mean is True:
        mean = np.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]
    if std is True:
        std = np.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]
    if mean is not None or std is not None:
        auglist.append(DetBorrowAug(ColorNormalizeAug(mean, std)))
    return auglist