import warnings
import numpy as np
from .activations import Activation
from ..block import Block, HybridBlock
from ..utils import _indent
from ... import nd, sym
from ...util import is_np_array
class HybridLambda(HybridBlock):
    """Wraps an operator or an expression as a HybridBlock object.

    Parameters
    ----------
    function : str or function
        Function used in lambda must be one of the following:
        1) The name of an operator that is available in both symbol and ndarray. For example::

            block = HybridLambda('tanh')

        2) A function that conforms to ``def function(F, data, *args)``. For example::

            block = HybridLambda(lambda F, x: F.LeakyReLU(x, slope=0.1))

    Inputs:
        - ** *args **: one or more input data. First argument must be symbol or ndarray. Their \\
            shapes depend on the function.

    Output:
        - ** *outputs **: one or more output data. Their shapes depend on the function.

    """

    def __init__(self, function, prefix=None):
        super(HybridLambda, self).__init__(prefix=prefix)
        if isinstance(function, str):
            assert hasattr(nd, function) and hasattr(sym, function), 'Function name %s is not found in symbol/ndarray.' % function
            func_dict = {sym: getattr(sym, function), nd: getattr(nd, function)}
            self._func = lambda F, *args: func_dict[F](*args)
            self._func_name = function
        elif callable(function):
            self._func = function
            self._func_name = function.__name__
        else:
            raise ValueError('Unrecognized function in lambda: {} of type {}'.format(function, type(function)))

    def hybrid_forward(self, F, x, *args):
        return self._func(F, x, *args)

    def __repr__(self):
        return '{name}({function})'.format(name=self.__class__.__name__, function=self._func_name)