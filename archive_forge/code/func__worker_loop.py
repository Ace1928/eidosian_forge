from the disk or network on-the-fly, without loading your entire corpus into RAM.
from __future__ import division  # py3 "true division"
import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from types import GeneratorType
import threading
import itertools
import copy
from queue import Queue, Empty
from numpy import float32 as REAL
import numpy as np
from gensim.utils import keep_vocab_item, call_on_class_only, deprecated
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
from gensim import utils, matutils
from gensim.models.keyedvectors import Vocab  # noqa
from smart_open.compression import get_supported_extensions
def _worker_loop(self, job_queue, progress_queue):
    """Train the model, lifting batches of data from the queue.

        This function will be called in parallel by multiple workers (threads or processes) to make
        optimal use of multicore machines.

        Parameters
        ----------
        job_queue : Queue of (list of objects, float)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented by a tuple where the first element is the corpus chunk to be processed and
            the second is the floating-point learning rate.
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * Size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
    thread_private_mem = self._get_thread_working_mem()
    jobs_processed = 0
    while True:
        job = job_queue.get()
        if job is None:
            progress_queue.put(None)
            break
        data_iterable, alpha = job
        tally, raw_tally = self._do_train_job(data_iterable, alpha, thread_private_mem)
        progress_queue.put((len(data_iterable), tally, raw_tally))
        jobs_processed += 1
    logger.debug('worker exiting, processed %i jobs', jobs_processed)