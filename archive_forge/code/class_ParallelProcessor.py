from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
class ParallelProcessor(SequentialProcessor):
    """
    Processor class for parallel processing of data.

    Parameters
    ----------
    processors : list
        Processor instances to be processed in parallel.
    num_threads : int, optional
        Number of parallel working threads.

    Notes
    -----
    If the `processors` list contains lists or tuples, these get wrapped as a
    :class:`SequentialProcessor`.

    """

    def __init__(self, processors, num_threads=None):
        super(ParallelProcessor, self).__init__(processors)
        if num_threads is None:
            num_threads = 1
        self.map = map
        if min(len(processors), max(1, num_threads)) > 1:
            self.map = mp.Pool(num_threads).map

    def process(self, data, **kwargs):
        """
        Process the data in parallel.

        Parameters
        ----------
        data : depends on the processors
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        list
            Processed data.

        """
        if len(self.processors) == 1:
            return [_process((self.processors[0], data, kwargs))]
        return list(self.map(_process, zip(self.processors, it.repeat(data), it.repeat(kwargs))))