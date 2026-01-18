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
def _assign_binary_codes(wv):
    """
    Appends a binary code to each vocab term.

    Parameters
    ----------
    wv : KeyedVectors
        A collection of word-vectors.

    Sets the .code and .point attributes of each node.
    Each code is a numpy.array containing 0s and 1s.
    Each point is an integer.

    """
    logger.info('constructing a huffman tree from %i words', len(wv))
    heap = _build_heap(wv)
    if not heap:
        logger.info('built huffman tree with maximum node depth 0')
        return
    max_depth = 0
    stack = [(heap[0], [], [])]
    while stack:
        node, codes, points = stack.pop()
        if node[1] < len(wv):
            k = node[1]
            wv.set_vecattr(k, 'code', codes)
            wv.set_vecattr(k, 'point', points)
            max_depth = max(len(codes), max_depth)
        else:
            points = np.array(list(points) + [node.index - len(wv)], dtype=np.uint32)
            stack.append((node.left, np.array(list(codes) + [0], dtype=np.uint8), points))
            stack.append((node.right, np.array(list(codes) + [1], dtype=np.uint8), points))
    logger.info('built huffman tree with maximum node depth %i', max_depth)