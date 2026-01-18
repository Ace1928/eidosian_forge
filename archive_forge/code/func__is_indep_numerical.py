import warnings
import numpy as np
from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode
from pennylane import numpy as pnp
def _is_indep_numerical(func, interface, args, kwargs, num_pos, seed, atol, rtol, bounds):
    """Test whether a function returns the same output at random positions.

    Args:
        func (callable): Function to be tested
        interface (str): Interface used by ``func``
        args (tuple): Positional arguments with respect to which to test
        kwargs (dict): Keyword arguments for ``func`` at which to test;
            the ``kwargs`` are kept fixed in this test.
        num_pos (int): Number of random positions to test
        seed (int): Seed for random number generator
        atol (float): Absolute tolerance for comparing the outputs
        rtol (float): Relative tolerance for comparing the outputs
        bounds (tuple[int, int]): Limits of the range from which to sample

    Returns:
        bool: Whether ``func`` returns the same output at the randomly
        chosen points.
    """
    rnd_args = _get_random_args(args, interface, num_pos, seed, bounds)
    original_output = func(*args, **kwargs)
    is_tuple_valued = isinstance(original_output, tuple)
    for _rnd_args in rnd_args:
        new_output = func(*_rnd_args, **kwargs)
        if is_tuple_valued:
            if not all((np.allclose(new, orig, atol=atol, rtol=rtol) for new, orig in zip(new_output, original_output))):
                return False
        elif not np.allclose(new_output, original_output, atol=atol, rtol=rtol):
            return False
    return True