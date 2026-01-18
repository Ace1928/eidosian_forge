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
def _log_epoch_progress(self, progress_queue=None, job_queue=None, cur_epoch=0, total_examples=None, total_words=None, report_delay=1.0, is_corpus_file_mode=None):
    """Get the progress report for a single training epoch.

        Parameters
        ----------
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.
        job_queue : Queue of (list of object, float)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented by a tuple where the first element is the corpus chunk to be processed and
            the second is the floating-point learning rate.
        cur_epoch : int, optional
            The current training epoch, needed to compute the training parameters for each job.
            For example in many implementations the learning rate would be dropping with the number of epochs.
        total_examples : int, optional
            Count of objects in the `data_iterator`. In the usual case this would correspond to the number of sentences
            in a corpus. Used to log progress.
        total_words : int, optional
            Count of total objects in `data_iterator`. In the usual case this would correspond to the number of raw
            words in a corpus. Used to log progress.
        report_delay : float, optional
            Number of seconds between two consecutive progress report messages in the logger.
        is_corpus_file_mode : bool, optional
            Whether training is file-based (corpus_file argument) or not.

        Returns
        -------
        (int, int, int)
            The epoch report consisting of three elements:
                * size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
    example_count, trained_word_count, raw_word_count = (0, 0, 0)
    start, next_report = (default_timer() - 1e-05, 1.0)
    job_tally = 0
    unfinished_worker_count = self.workers
    while unfinished_worker_count > 0:
        report = progress_queue.get()
        if report is None:
            unfinished_worker_count -= 1
            logger.debug('worker thread finished; awaiting finish of %i more threads', unfinished_worker_count)
            continue
        examples, trained_words, raw_words = report
        job_tally += 1
        example_count += examples
        trained_word_count += trained_words
        raw_word_count += raw_words
        elapsed = default_timer() - start
        if elapsed >= next_report:
            self._log_progress(job_queue, progress_queue, cur_epoch, example_count, total_examples, raw_word_count, total_words, trained_word_count, elapsed)
            next_report = elapsed + report_delay
    elapsed = default_timer() - start
    self._log_epoch_end(cur_epoch, example_count, total_examples, raw_word_count, total_words, trained_word_count, elapsed, is_corpus_file_mode)
    self.total_train_time += elapsed
    return (trained_word_count, raw_word_count, job_tally)