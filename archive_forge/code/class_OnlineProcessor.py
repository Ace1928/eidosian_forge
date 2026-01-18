from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
class OnlineProcessor(Processor):
    """
    Abstract base class for processing data in online mode.

    Derived classes must implement the following methods:

    - process_online(): process the data in online mode,
    - process_offline(): process the data in offline mode.

    """

    def __init__(self, online=False):
        self.online = online

    def process(self, data, **kwargs):
        """
        Process the data either in online or offline mode.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        Notes
        -----
        This method is used to pass the data to either `process_online` or
        `process_offline`, depending on the `online` setting of the processor.

        """
        if self.online:
            return self.process_online(data, **kwargs)
        return self.process_offline(data, **kwargs)

    def process_online(self, data, reset=True, **kwargs):
        """
        Process the data in online mode.

        This method must be implemented by the derived class and should process
        the given data frame by frame and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        reset : bool, optional
            Reset the processor to its initial state before processing.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def process_offline(self, data, **kwargs):
        """
        Process the data in offline mode.

        This method must be implemented by the derived class and should process
        the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def reset(self):
        """
        Reset the OnlineProcessor.

        This method must be implemented by the derived class and should reset
        the processor to its initial state.

        """
        raise NotImplementedError('Must be implemented by subclass.')