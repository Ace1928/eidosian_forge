import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def _handle_input_3D(image, footprint=None, out=None, mask=None, out_dtype=None, pixel_size=1, shift_x=None, shift_y=None, shift_z=None):
    """Preprocess and verify input for filters.rank methods.

    Parameters
    ----------
    image : 3-D array (integer or float)
        Input image.
    footprint : 3-D array (integer or float), optional
        The neighborhood expressed as a 3-D array of 1's and 0's.
    out : 3-D array (integer or float), optional
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel. Default value is 1.
    shift_x, shift_y, shift_z : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint size (center must be inside of the given footprint).

    Returns
    -------
    image : 3-D array (np.uint8 or np.uint16)
    footprint : 3-D array (np.uint8)
        The neighborhood expressed as a binary 3-D array.
    out : 3-D array (same dtype out_dtype or as input)
        Output array. The two first dimensions are the spatial ones, the third
        one is the pixel vector (length 1 by default).
    mask : 3-D array (np.uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood.
    n_bins : int
        Number of histogram bins.

    """
    check_nD(image, 3)
    if image.dtype not in (np.uint8, np.uint16):
        message = f'Possible precision loss converting image of type {image.dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.'
        warn(message, stacklevel=2)
        image = img_as_ubyte(image)
    footprint = np.ascontiguousarray(img_as_ubyte(footprint > 0))
    if footprint.ndim != image.ndim:
        raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image = np.ascontiguousarray(image)
    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = img_as_ubyte(mask)
        mask = np.ascontiguousarray(mask)
    if image is out:
        raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:
        if out_dtype is None:
            out_dtype = image.dtype
        out = np.empty(image.shape + (pixel_size,), dtype=out_dtype)
    else:
        out = out.reshape(out.shape + (pixel_size,))
    is_8bit = image.dtype in (np.uint8, np.int8)
    if is_8bit:
        n_bins = 256
    else:
        n_bins = int(max(3, image.max())) + 1
    if n_bins > 2 ** 10:
        warn(f'Bad rank filter performance is expected due to a large number of bins ({n_bins}), equivalent to an approximate bitdepth of {np.log2(n_bins):.1f}.', stacklevel=2)
    for name, value in zip(('shift_x', 'shift_y', 'shift_z'), (shift_x, shift_y, shift_z)):
        if np.dtype(type(value)) == bool:
            warn(f'Parameter `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.', category=UserWarning, stacklevel=4)
    return (image, footprint, out, mask, n_bins)