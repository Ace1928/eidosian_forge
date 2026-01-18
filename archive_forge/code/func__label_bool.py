from scipy import ndimage
from ._ccomp import label_cython as clabel
def _label_bool(image, background=None, return_num=False, connectivity=None):
    """Faster implementation of clabel for boolean input.

    See context: https://github.com/scikit-image/scikit-image/issues/4833
    """
    from ..morphology._util import _resolve_neighborhood
    if background == 1:
        image = ~image
    if connectivity is None:
        connectivity = image.ndim
    if not 1 <= connectivity <= image.ndim:
        raise ValueError(f'Connectivity for {image.ndim}D image should be in [1, ..., {image.ndim}]. Got {connectivity}.')
    footprint = _resolve_neighborhood(None, connectivity, image.ndim)
    result = ndimage.label(image, structure=footprint)
    if return_num:
        return result
    else:
        return result[0]