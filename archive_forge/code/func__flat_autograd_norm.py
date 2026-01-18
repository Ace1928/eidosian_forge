import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
def _flat_autograd_norm(tensor, **kwargs):
    """Helper function for computing the norm of an autograd tensor when the order or axes are not
    specified. This is used for differentiability."""
    x = np.ravel(tensor)
    sq_norm = np.dot(x, np.conj(x))
    return np.real(np.sqrt(sq_norm))