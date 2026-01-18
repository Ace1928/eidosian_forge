from itertools import product
import numpy as np
Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    This function computes the coefficients blindly without any filtering applied, and
    is thus used as a helper function for the true ``coefficients`` function.

    Args:
        f (callable): function that takes a 1D array of scalar inputs
        degree (int or tuple[int]): max frequency of Fourier coeffs to be computed. For degree
            :math:`d`, the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d`
            will be computed.
        use_broadcasting (bool): Whether or not to broadcast the parameters to execute
            multiple function calls at once. Broadcasting is performed along the last axis
            of the grid of evaluation points.

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.
    