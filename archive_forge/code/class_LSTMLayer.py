from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class LSTMLayer(RecurrentLayer):
    """
    Recurrent network layer with Long Short-Term Memory units.

    Parameters
    ----------
    input_gate : :class:`Gate`
        Input gate.
    forget_gate : :class:`Gate`
        Forget gate.
    cell : :class:`Cell`
        Cell (i.e. a Gate without peephole connections).
    output_gate : :class:`Gate`
        Output gate.
    activation_fn : numpy ufunc, optional
        Activation function.
    init : numpy array, shape (num_hiddens, ), optional
        Initial state of the layer.
    cell_init : numpy array, shape (num_hiddens, ), optional
        Initial state of the cell.

    """

    def __init__(self, input_gate, forget_gate, cell, output_gate, activation_fn=tanh, init=None, cell_init=None):
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell = cell
        self.output_gate = output_gate
        self.activation_fn = activation_fn
        if init is None:
            init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init
        if cell_init is None:
            cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.cell_init = cell_init
        self._state = self.cell_init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_prev', None)
        state.pop('_state', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'init'):
            self.init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        if not hasattr(self, 'cell_init'):
            self.cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self._prev = self.init
        self._state = self.cell_init

    def reset(self, init=None, cell_init=None):
        """
        Reset the layer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.
        cell_init : numpy array, shape (num_hiddens,), optional
            Reset the cells to this initial state.

        """
        self._prev = init if init is not None else self.init
        self._state = cell_init if cell_init is not None else self.cell_init

    def activate(self, data, reset=True):
        """
        Activate the LSTM layer.

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
            ig = self.input_gate.activate(data_, self._prev, self._state)
            fg = self.forget_gate.activate(data_, self._prev, self._state)
            cell = self.cell.activate(data_, self._prev)
            self._state = cell * ig + self._state * fg
            og = self.output_gate.activate(data_, self._prev, self._state)
            out[i] = self.activation_fn(self._state) * og
            self._prev = out[i]
        return out