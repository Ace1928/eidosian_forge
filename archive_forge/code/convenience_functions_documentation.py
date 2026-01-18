from typing import Callable, List, Tuple, Union
import numpy as np

    Decorates a smooth function, creating a piece-wise constant function that approximates it.

    Creates a callable for defining a :class:`~.ParametrizedHamiltonian`.

    Args:
        timespan(Union[float, tuple(float)]): The time span defining the region where the function is non-zero.
            If a ``float`` is provided, the time span is defined as ``(0, timespan)``.
        num_bins(int): number of bins for time-binning the function

    Returns:
        callable: a function that takes some smooth function ``f(params, t)`` and converts it to a
        piece-wise constant function spanning time ``t`` in ``num_bins`` bins.

    **Example**

    .. code-block:: python3

        def smooth_function(params, t):
            return params[0] * t + params[1]

        timespan = 10
        num_bins = 10

        binned_function = qml.pulse.pwc_from_function(timespan, num_bins)(smooth_function)

    >>> binned_function([2, 4], 3), smooth_function([2, 4], 3)  # t = 3
    (Array(10.666667, dtype=float32), 10)

    >>> binned_function([2, 4], 3.2), smooth_function([2, 4], 3.2)  # t = 3.2
    (Array(10.666667, dtype=float32), 10.4)

    >>> binned_function([2, 4], 4.5), smooth_function([2, 4], 4.5)  # t = 4.5
    (Array(12.888889, dtype=float32), 13.0)

    The same effect can be achieved by decorating the smooth function:

    .. code-block:: python

        from pennylane.pulse.convenience_functions import pwc_from_function

        @pwc_from_function(timespan, num_bins)
        def fn(params, t):
            return params[0] * t + params[1]

    >>> fn([2, 4], 3)
    Array(10.666667, dtype=float32)

    