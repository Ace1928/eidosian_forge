import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class channel_as_last_axis:
    """Decorator for automatically making channels axis last for all arrays.

    This decorator reorders axes for compatibility with functions that only
    support channels along the last axis. After the function call is complete
    the channels axis is restored back to its original position.

    Parameters
    ----------
    channel_arg_positions : tuple of int, optional
        Positional arguments at the positions specified in this tuple are
        assumed to be multichannel arrays. The default is to assume only the
        first argument to the function is a multichannel array.
    channel_kwarg_names : tuple of str, optional
        A tuple containing the names of any keyword arguments corresponding to
        multichannel arrays.
    multichannel_output : bool, optional
        A boolean that should be True if the output of the function is not a
        multichannel array and False otherwise. This decorator does not
        currently support the general case of functions with multiple outputs
        where some or all are multichannel.

    """

    def __init__(self, channel_arg_positions=(0,), channel_kwarg_names=(), multichannel_output=True):
        self.arg_positions = set(channel_arg_positions)
        self.kwarg_names = set(channel_kwarg_names)
        self.multichannel_output = multichannel_output

    def __call__(self, func):

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            channel_axis = kwargs.get('channel_axis', None)
            if channel_axis is None:
                return func(*args, **kwargs)
            if np.isscalar(channel_axis):
                channel_axis = (channel_axis,)
            if len(channel_axis) > 1:
                raise ValueError('only a single channel axis is currently supported')
            if channel_axis == (-1,) or channel_axis == -1:
                return func(*args, **kwargs)
            if self.arg_positions:
                new_args = []
                for pos, arg in enumerate(args):
                    if pos in self.arg_positions:
                        new_args.append(np.moveaxis(arg, channel_axis[0], -1))
                    else:
                        new_args.append(arg)
                new_args = tuple(new_args)
            else:
                new_args = args
            for name in self.kwarg_names:
                kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)
            kwargs['channel_axis'] = -1
            out = func(*new_args, **kwargs)
            if self.multichannel_output:
                out = np.moveaxis(out, -1, channel_axis[0])
            return out
        return fixed_func