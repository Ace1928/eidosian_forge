from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
class _ParallelProcess(mp.Process):
    """
    Class for processing tasks in a queue.

    Parameters
    ----------
    task_queue :
        Queue with tasks, i.e. tuples ('processor', 'infile', 'outfile')

    Notes
    -----
    Usually, multiple instances are created via :func:`process_batch`.

    """

    def __init__(self, task_queue):
        super(_ParallelProcess, self).__init__()
        self.task_queue = task_queue

    def run(self):
        """Process all tasks from the task queue."""
        from .audio.signal import LoadAudioFileError
        while True:
            processor, infile, outfile, kwargs = self.task_queue.get()
            try:
                _process((processor, infile, outfile, kwargs))
            except LoadAudioFileError as e:
                print(e)
            self.task_queue.task_done()