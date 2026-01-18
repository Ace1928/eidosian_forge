import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
class ResolutionTooDeep(ResolutionError):

    def __init__(self, round_count):
        super(ResolutionTooDeep, self).__init__(round_count)
        self.round_count = round_count