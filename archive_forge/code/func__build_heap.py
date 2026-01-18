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
def _build_heap(wv):
    heap = list((Heapitem(wv.get_vecattr(i, 'count'), i, None, None) for i in range(len(wv.index_to_key))))
    heapq.heapify(heap)
    for i in range(len(wv) - 1):
        min1, min2 = (heapq.heappop(heap), heapq.heappop(heap))
        heapq.heappush(heap, Heapitem(count=min1.count + min2.count, index=i + len(wv), left=min1, right=min2))
    return heap