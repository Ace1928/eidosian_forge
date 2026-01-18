from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class GRULayer(RecurrentLayer):
    """
    Recurrent network layer with Gated Recurrent Units (GRU) as proposed in
    [1]_.

    Parameters
    ----------
    reset_gate : :class:`Gate`
        Reset gate.
    update_gate : :class:`Gate`
        Update gate.
    cell : :class:`GRUCell`
        GRU cell.
    init : numpy array, shape (num_hiddens,), optional
        Initial state of hidden units.

    References
    ----------
    .. [1] Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, and Yoshua
           Bengio,
           "On the properties of neural machine translation: Encoder-decoder
           approaches",
           http://arxiv.org/abs/1409.1259, 2014.

    Notes
    -----
    There are two formulations of the GRUCell in the literature. Here,
    we adopted the (slightly older) one proposed in [1], which is also
    implemented in the Lasagne toolbox.

    """

    def __init__(self, reset_gate, update_gate, cell, init=None):
        self.reset_gate = reset_gate
        self.update_gate = update_gate
        self.cell = cell
        if init is None:
            init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_prev', None)
        return state

    def __setstate__(self, state):
        try:
            import warnings
            warnings.warn('Please update your GRU models by loading them and saving them again. Loading old models will not work from version 0.18 onwards.', RuntimeWarning)
            state['init'] = state.pop('hid_init')
        except KeyError:
            pass
        self.__dict__.update(state)
        if not hasattr(self, 'init'):
            self.init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self._prev = self.init

    def reset(self, init=None):
        """
        Reset the layer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.

        """
        self._prev = init or self.init

    def activate(self, data, reset=True):
        """
        Activate the GRU layer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.
        reset : bool, optional
            Reset the layer to its initial state before activating it.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        if reset:
            self.reset()
        size = len(data)
        out = np.zeros((size, self.cell.bias.size), dtype=NN_DTYPE)
        for i in range(size):
            data_ = data[i]
            rg = self.reset_gate.activate(data_, self._prev)
            ug = self.update_gate.activate(data_, self._prev)
            cell = self.cell.activate(data_, self._prev, rg)
            out[i] = ug * cell + (1 - ug) * self._prev
            self._prev = out[i]
        return out