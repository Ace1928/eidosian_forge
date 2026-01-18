from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _get_activation(self, F, inputs, activation, **kwargs):
    """Get activation function. Convert if is string"""
    func = {'tanh': F.tanh, 'relu': F.relu, 'sigmoid': F.sigmoid, 'softsign': F.softsign}.get(activation)
    if func:
        return func(inputs, **kwargs)
    elif isinstance(activation, string_types):
        return F.Activation(inputs, act_type=activation, **kwargs)
    elif isinstance(activation, LeakyReLU):
        return F.LeakyReLU(inputs, act_type='leaky', slope=activation._alpha, **kwargs)
    return activation(inputs, **kwargs)