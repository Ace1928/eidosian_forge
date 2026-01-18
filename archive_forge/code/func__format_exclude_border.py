import math
import numpy as np
import scipy.ndimage as ndi
from scipy import spatial
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max
def _format_exclude_border(img_ndim, exclude_border):
    """Format an ``exclude_border`` argument as a tuple of ints for calling
    ``peak_local_max``.
    """
    if isinstance(exclude_border, tuple):
        if len(exclude_border) != img_ndim:
            raise ValueError('`exclude_border` should have the same length as the dimensionality of the image.')
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError('exclude border, when expressed as a tuple, must only contain ints.')
        return exclude_border + (0,)
    elif isinstance(exclude_border, int):
        return (exclude_border,) * img_ndim + (0,)
    elif exclude_border is True:
        raise ValueError('exclude_border cannot be True')
    elif exclude_border is False:
        return (0,) * (img_ndim + 1)
    else:
        raise ValueError(f'Unsupported value ({exclude_border}) for exclude_border')