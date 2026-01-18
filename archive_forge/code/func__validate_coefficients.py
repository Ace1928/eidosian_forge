from itertools import product
import numpy as np
from .utils import format_nvec
def _validate_coefficients(coeffs, n_inputs, can_be_list=True):
    """Helper function to validate input coefficients of plotting functions.

    Args:
        coeffs (array[complex]): A set (or list of sets) of Fourier coefficients of a
            n_inputs-dimensional function.
        n_inputs (int): The number of inputs (dimension) of the function the coefficients are for.
        can_be_list (bool): Whether or not the plotting function accepts a list of
            coefficients, or only a single set.

    Raises:
        TypeError: If the coefficients are not a list or array.
        ValueError: if the coefficients are not a suitable type for the plotting function.
    """
    if not isinstance(coeffs, list) and (not isinstance(coeffs, np.ndarray)):
        raise TypeError(f'Input to coefficient plotting functions must be a list of numerical Fourier coefficients. Received input of type {type(coeffs)}')
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)
    if len(coeffs.shape) == n_inputs and can_be_list:
        coeffs = np.array([coeffs])
    required_shape_size = n_inputs + 1 if can_be_list else n_inputs
    if len(coeffs.shape) != required_shape_size:
        raise ValueError(f'Plotting function expected a list of {n_inputs}-dimensional inputs. Received coefficients of {len(coeffs.shape)}-dimensional function.')
    dims = coeffs.shape[1:] if can_be_list else coeffs.shape
    if any(((dim - 1) % 2 for dim in dims)):
        raise ValueError(f'Shape of input coefficients must be 2d_i + 1, where d_i is the largest frequency in the i-th input. Coefficient array with shape {coeffs.shape} is invalid.')
    return coeffs