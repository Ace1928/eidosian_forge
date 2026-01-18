import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray
from . util import is_np_array
from . import numpy as _mx_np  # pylint: disable=reimported
@register
class FusedRNN(Initializer):
    """Initialize parameters for fused rnn layers.

    Parameters
    ----------
    init : Initializer
        initializer applied to unpacked weights. Fall back to global
        initializer if None.
    num_hidden : int
        should be the same with arguments passed to FusedRNNCell.
    num_layers : int
        should be the same with arguments passed to FusedRNNCell.
    mode : str
        should be the same with arguments passed to FusedRNNCell.
    bidirectional : bool
        should be the same with arguments passed to FusedRNNCell.
    forget_bias : float
        should be the same with arguments passed to FusedRNNCell.
    """

    def __init__(self, init, num_hidden, num_layers, mode, bidirectional=False, forget_bias=1.0):
        if isinstance(init, string_types):
            klass, kwargs = json.loads(init)
            init = registry._REGISTRY[klass.lower()](**kwargs)
        super(FusedRNN, self).__init__(init=init.dumps() if init is not None else None, num_hidden=num_hidden, num_layers=num_layers, mode=mode, bidirectional=bidirectional, forget_bias=forget_bias)
        self._init = init
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._mode = mode
        self._bidirectional = bidirectional
        self._forget_bias = forget_bias

    def _init_weight(self, desc, arr):
        from .rnn import rnn_cell
        cell = rnn_cell.FusedRNNCell(self._num_hidden, self._num_layers, self._mode, self._bidirectional, forget_bias=self._forget_bias, prefix='')
        args = cell.unpack_weights({'parameters': arr})
        for name in args:
            arg_desc = InitDesc(name, global_init=desc.global_init)
            if self._mode == 'lstm' and name.endswith('_f_bias'):
                args[name][:] = self._forget_bias
            elif self._init is None:
                desc.global_init(arg_desc, args[name])
            else:
                self._init(arg_desc, args[name])
        arr[:] = cell.pack_weights(args)['parameters']