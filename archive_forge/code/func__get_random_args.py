import warnings
import numpy as np
from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode
from pennylane import numpy as pnp
def _get_random_args(args, interface, num, seed, bounds):
    """Generate random arguments of a given structure.

    Args:
        args (tuple): Original input arguments
        interface (str): Interface of the QNode into which the arguments will be fed
        num (int): Number of random argument sets to generate
        seed (int): Seed for random generation
        bounds (tuple[int]): Range within which to sample the random parameters.

    Returns:
        list[tuple]: List of length ``num`` with each entry being a random instance
        of arguments like ``args``.

    This function generates ``num`` many tuples of random arguments in the given range
    that have the same shapes as ``args``.
    """
    width = bounds[1] - bounds[0]
    if interface == 'tf':
        import tensorflow as tf
        tf.random.set_seed(seed)
        rnd_args = []
        for _ in range(num):
            _args = (tf.random.uniform(tf.shape(_arg)) * width + bounds[0] for _arg in args)
            _args = tuple((tf.Variable(_arg) if isinstance(arg, tf.Variable) else _arg for _arg, arg in zip(_args, args)))
            rnd_args.append(_args)
    elif interface == 'torch':
        import torch
        torch.random.manual_seed(seed)
        rnd_args = [tuple((torch.rand(np.shape(arg)) * width + bounds[0] for arg in args)) for _ in range(num)]
    else:
        rng = np.random.default_rng(seed)
        rnd_args = [tuple((rng.random(np.shape(arg)) * width + bounds[0] for arg in args)) for _ in range(num)]
        if interface == 'autograd':
            rnd_args = [tuple((pnp.array(a, requires_grad=True) for a in arg)) for arg in rnd_args]
    return rnd_args