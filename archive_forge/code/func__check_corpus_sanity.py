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
def _check_corpus_sanity(self, corpus_iterable=None, corpus_file=None, passes=1):
    """Checks whether the corpus parameters make sense."""
    if corpus_file is None and corpus_iterable is None:
        raise TypeError('Either one of corpus_file or corpus_iterable value must be provided')
    if corpus_file is not None and corpus_iterable is not None:
        raise TypeError('Both corpus_file and corpus_iterable must not be provided at the same time')
    if corpus_iterable is None and (not os.path.isfile(corpus_file)):
        raise TypeError('Parameter corpus_file must be a valid path to a file, got %r instead' % corpus_file)
    if corpus_iterable is not None and (not isinstance(corpus_iterable, Iterable)):
        raise TypeError('The corpus_iterable must be an iterable of lists of strings, got %r instead' % corpus_iterable)
    if corpus_iterable is not None and isinstance(corpus_iterable, GeneratorType) and (passes > 1):
        raise TypeError(f"Using a generator as corpus_iterable can't support {passes} passes. Try a re-iterable sequence.")
    if corpus_iterable is None:
        _, corpus_ext = os.path.splitext(corpus_file)
        if corpus_ext.lower() in get_supported_extensions():
            raise TypeError(f'Training from compressed files is not supported with the `corpus_path` argument. Please decompress {corpus_file} or use `corpus_iterable` instead.')