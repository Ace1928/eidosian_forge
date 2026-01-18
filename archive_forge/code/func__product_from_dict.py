import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float
def _product_from_dict(dictionary):
    """Utility function to convert parameter ranges to parameter combinations.

    Converts a dict of lists into a list of dicts whose values consist of the
    cartesian product of the values in the original dict.

    Parameters
    ----------
    dictionary : dict of lists
        Dictionary of lists to be multiplied.

    Yields
    ------
    selections : dicts of values
        Dicts containing individual combinations of the values in the input
        dict.
    """
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))