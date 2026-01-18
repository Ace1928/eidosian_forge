from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
class PickleView(Table):

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        with self.source.open('rb') as f:
            try:
                while True:
                    yield tuple(pickle.load(f))
            except EOFError:
                pass