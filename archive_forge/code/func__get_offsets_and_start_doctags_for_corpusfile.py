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
@classmethod
def _get_offsets_and_start_doctags_for_corpusfile(cls, corpus_file, workers):
    """Get offset and initial document tag in a corpus_file for each worker.

        Firstly, approximate offsets are calculated based on number of workers and corpus_file size.
        Secondly, for each approximate offset we find the maximum offset which points to the beginning of line and
        less than approximate offset.

        Parameters
        ----------
        corpus_file : str
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
        workers : int
            Number of workers.

        Returns
        -------
        list of int, list of int
            Lists with offsets and document tags with length = number of workers.
        """
    corpus_file_size = os.path.getsize(corpus_file)
    approx_offsets = [int(corpus_file_size // workers * i) for i in range(workers)]
    offsets = []
    start_doctags = []
    with utils.open(corpus_file, mode='rb') as fin:
        curr_offset_idx = 0
        prev_filepos = 0
        for line_no, line in enumerate(fin):
            if curr_offset_idx == len(approx_offsets):
                break
            curr_filepos = prev_filepos + len(line)
            while curr_offset_idx != len(approx_offsets) and approx_offsets[curr_offset_idx] < curr_filepos:
                offsets.append(prev_filepos)
                start_doctags.append(line_no)
                curr_offset_idx += 1
            prev_filepos = curr_filepos
    return (offsets, start_doctags)