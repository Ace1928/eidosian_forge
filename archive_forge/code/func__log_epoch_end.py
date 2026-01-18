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
def _log_epoch_end(self, cur_epoch, example_count, total_examples, raw_word_count, total_words, trained_word_count, elapsed, is_corpus_file_mode):
    """Callback used to log the end of a training epoch.

        Parameters
        ----------
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
        is_corpus_file_mode : bool
            Whether training is file-based (corpus_file argument) or not.

        Warnings
        --------
        In case the corpus is changed while the epoch was running.

        """
    logger.info('EPOCH %i: training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s', cur_epoch, raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
    if is_corpus_file_mode:
        return
    if total_examples and total_examples != example_count:
        logger.warning('EPOCH %i: supplied example count (%i) did not equal expected count (%i)', cur_epoch, example_count, total_examples)
    if total_words and total_words != raw_word_count:
        logger.warning('EPOCH %i: supplied raw word count (%i) did not equal expected count (%i)', cur_epoch, raw_word_count, total_words)