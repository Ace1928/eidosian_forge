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
class _NamedTempFileDeleteOnGC(object):

    def __init__(self, name):
        self.name = name

    def delete(self, unlink=os.unlink, log=logger.debug):
        name = self.name
        try:
            log('deleting %s' % name)
            unlink(name)
        except Exception as e:
            log('exception deleting %s: %s' % (name, e))
            raise
        else:
            log('deleted %s' % name)

    def __del__(self):
        self.delete()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name