from __future__ import absolute_import, print_function, division
import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type
import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices
def _shortlistmergesorted(key=None, reverse=False, *iterables):
    """Return a single iterator over the given iterables, sorted by the
    given `key` function, assuming the input iterables are already sorted by
    the same function. (I.e., the merge part of a general merge sort.) Uses
    :func:`min` (or :func:`max` if ``reverse=True``) for the underlying
    implementation."""
    if reverse:
        op = max
    else:
        op = min
    if key is not None:
        opkwargs = {'key': key}
    else:
        opkwargs = dict()
    iterators = list()
    shortlist = list()
    for iterable in iterables:
        it = iter(iterable)
        try:
            first = next(it)
            iterators.append(it)
            shortlist.append(first)
        except StopIteration:
            pass
    while iterators:
        nxt = op(shortlist, **opkwargs)
        yield nxt
        nextidx = shortlist.index(nxt)
        try:
            shortlist[nextidx] = next(iterators[nextidx])
        except StopIteration:
            del shortlist[nextidx]
            del iterators[nextidx]