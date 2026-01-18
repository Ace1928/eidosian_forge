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
def _scan_vocab(self, corpus_iterable, progress_per, trim_rule):
    document_no = -1
    total_words = 0
    min_reduce = 1
    interval_start = default_timer() - 1e-05
    interval_count = 0
    checked_string_types = 0
    vocab = defaultdict(int)
    max_rawint = -1
    doctags_lookup = {}
    doctags_list = []
    for document_no, document in enumerate(corpus_iterable):
        if not checked_string_types:
            if isinstance(document.words, str):
                logger.warning("Each 'words' should be a list of words (usually unicode strings). First 'words' here is instead plain %s.", type(document.words))
            checked_string_types += 1
        if document_no % progress_per == 0:
            interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
            logger.info('PROGRESS: at example #%i, processed %i words (%i words/s), %i word types, %i tags', document_no, total_words, interval_rate, len(vocab), len(doctags_list))
            interval_start = default_timer()
            interval_count = total_words
        document_length = len(document.words)
        for tag in document.tags:
            if isinstance(tag, (int, integer)):
                max_rawint = max(max_rawint, tag)
            elif tag in doctags_lookup:
                doctags_lookup[tag].doc_count += 1
                doctags_lookup[tag].word_count += document_length
            else:
                doctags_lookup[tag] = Doctag(index=len(doctags_list), word_count=document_length, doc_count=1)
                doctags_list.append(tag)
        for word in document.words:
            vocab[word] += 1
        total_words += len(document.words)
        if self.max_vocab_size and len(vocab) > self.max_vocab_size:
            utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
            min_reduce += 1
    corpus_count = document_no + 1
    if len(doctags_list) > corpus_count:
        logger.warning('More unique tags (%i) than documents (%i).', len(doctags_list), corpus_count)
    if max_rawint > corpus_count:
        logger.warning('Highest int doctag (%i) larger than count of documents (%i). This means at least %i excess, unused slots (%i bytes) will be allocated for vectors.', max_rawint, corpus_count, max_rawint - corpus_count, (max_rawint - corpus_count) * self.vector_size * dtype(REAL).itemsize)
    if max_rawint > -1:
        for key in doctags_list:
            doctags_lookup[key].index = doctags_lookup[key].index + max_rawint + 1
        doctags_list = list(range(0, max_rawint + 1)) + doctags_list
    self.dv.index_to_key = doctags_list
    for t, dt in doctags_lookup.items():
        self.dv.key_to_index[t] = dt.index
        self.dv.set_vecattr(t, 'word_count', dt.word_count)
        self.dv.set_vecattr(t, 'doc_count', dt.doc_count)
    self.raw_vocab = vocab
    return (total_words, corpus_count)