import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def _do_train_epoch(self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch, total_examples=None, total_words=None, offsets=None, start_doctags=None, **kwargs):
    work, neu1 = thread_private_mem
    doctag_vectors = self.dv.vectors
    doctags_lockf = self.dv.vectors_lockf
    offset = offsets[thread_id]
    start_doctag = start_doctags[thread_id]
    if self.sg:
        examples, tally, raw_tally = d2v_train_epoch_dbow(self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch, total_examples, total_words, work, neu1, len(self.dv), doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf, train_words=self.dbow_words)
    elif self.dm_concat:
        examples, tally, raw_tally = d2v_train_epoch_dm_concat(self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch, total_examples, total_words, work, neu1, len(self.dv), doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
    else:
        examples, tally, raw_tally = d2v_train_epoch_dm(self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch, total_examples, total_words, work, neu1, len(self.dv), doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
    return (examples, tally, raw_tally)