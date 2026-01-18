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
def _train_epoch_corpusfile(self, corpus_file, cur_epoch=0, total_examples=None, total_words=None, callbacks=(), **kwargs):
    """Train the model for a single epoch.

        Parameters
        ----------
        corpus_file : str
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
        cur_epoch : int, optional
            The current training epoch, needed to compute the training parameters for each job.
            For example in many implementations the learning rate would be dropping with the number of epochs.
        total_examples : int, optional
            Count of objects in the `data_iterator`. In the usual case this would correspond to the number of sentences
            in a corpus, used to log progress.
        total_words : int
            Count of total objects in `data_iterator`. In the usual case this would correspond to the number of raw
            words in a corpus, used to log progress. Must be provided in order to seek in `corpus_file`.
        **kwargs : object
            Additional key word parameters for the specific model inheriting from this class.

        Returns
        -------
        (int, int, int)
            The training report for this epoch consisting of three elements:
                * Size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
    if not total_words:
        raise ValueError('total_words must be provided alongside corpus_file argument.')
    from gensim.models.word2vec_corpusfile import CythonVocab
    from gensim.models.fasttext import FastText
    cython_vocab = CythonVocab(self.wv, hs=self.hs, fasttext=isinstance(self, FastText))
    progress_queue = Queue()
    corpus_file_size = os.path.getsize(corpus_file)
    thread_kwargs = copy.copy(kwargs)
    thread_kwargs['cur_epoch'] = cur_epoch
    thread_kwargs['total_examples'] = total_examples
    thread_kwargs['total_words'] = total_words
    workers = [threading.Thread(target=self._worker_loop_corpusfile, args=(corpus_file, thread_id, corpus_file_size / self.workers * thread_id, cython_vocab, progress_queue), kwargs=thread_kwargs) for thread_id in range(self.workers)]
    for thread in workers:
        thread.daemon = True
        thread.start()
    trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(progress_queue=progress_queue, job_queue=None, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words, is_corpus_file_mode=True)
    return (trained_word_count, raw_word_count, job_tally)