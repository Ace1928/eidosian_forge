from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
class BufferProcessor(Processor):
    """
    Buffer for processors which need context to do their processing.

    Parameters
    ----------
    buffer_size : int or tuple
        Size of the buffer (time steps, [additional dimensions]).
    init : numpy array, optional
        Init the buffer with this array.
    init_value : float, optional
        If only `buffer_size` is given but no `init`, use this value to
        initialise the buffer.

    Notes
    -----
    If `buffer_size` (or the first item thereof in case of tuple) is 1,
    only the un-buffered current value is returned.

    If context is needed, `buffer_size` must be set to >1.
    E.g. SpectrogramDifference needs a context of two frames to be able to
    compute the difference between two consecutive frames.

    """

    def __init__(self, buffer_size=None, init=None, init_value=0):
        if buffer_size is None and init is not None:
            buffer_size = init.shape
        elif isinstance(buffer_size, integer_types):
            buffer_size = (buffer_size,)
        if buffer_size is not None and init is None:
            init = np.ones(buffer_size) * init_value
        self.buffer_size = buffer_size
        self.init = init
        self.data = init

    def reset(self, init=None):
        """
        Reset BufferProcessor to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset BufferProcessor to this initial state.

        """
        self.data = init if init is not None else self.init

    def process(self, data, **kwargs):
        """
        Buffer the data.

        Parameters
        ----------
        data : numpy array or subclass thereof
            Data to be buffered.

        Returns
        -------
        numpy array or subclass thereof
            Data with buffered context.

        """
        ndmin = len(self.buffer_size)
        if data.ndim < ndmin:
            data = np.array(data, copy=False, subok=True, ndmin=ndmin)
        data_length = len(data)
        self.data = np.roll(self.data, -data_length, axis=0)
        self.data[-data_length:] = data
        return self.data
    buffer = process

    def __getitem__(self, index):
        """
        Direct access to the buffer data.

        Parameters
        ----------
        index : int, slice, ndarray,
            Any NumPy indexing method to access the buffer data directly.

        Returns
        -------
        numpy array or subclass thereof
            Requested view of the buffered data.

        """
        return self.data[index]