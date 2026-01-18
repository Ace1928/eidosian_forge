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
def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count, total_examples, raw_word_count, total_words, trained_word_count, elapsed):
    """Callback used to log progress for long running jobs.

        Parameters
        ----------
        job_queue : Queue of (list of object, float)
            The queue of jobs still to be performed by workers. Each job is represented as a tuple containing
            the batch of data to be processed and the floating-point learning rate.
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.
        cur_epoch : int
            The current training iteration through the corpus.
        example_count : int
            Number of examples (could be sentences for example) processed until now.
        total_examples : int
            Number of all examples present in the input corpus.
        raw_word_count : int
            Number of words used in training until now.
        total_words : int
            Number of all words in the input corpus.
        trained_word_count : int
            Number of effective words used in training until now (after ignoring unknown words and trimming
            the sentence length).
        elapsed : int
            Elapsed time since the beginning of training in seconds.

        Notes
        -----
        If you train the model via `corpus_file` argument, there is no job_queue, so reported job_queue size will
        always be equal to -1.

        """
    if total_examples:
        logger.info('EPOCH %i - PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i', cur_epoch, 100.0 * example_count / total_examples, trained_word_count / elapsed, -1 if job_queue is None else utils.qsize(job_queue), utils.qsize(progress_queue))
    else:
        logger.info('EPOCH %i - PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i', cur_epoch, 100.0 * raw_word_count / total_words, trained_word_count / elapsed, -1 if job_queue is None else utils.qsize(job_queue), utils.qsize(progress_queue))