import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
class CyclicList4(list):

    def __init__(self):
        return list.__init__(self, [None, None, None, None])

    def __getitem__(self, n):
        return list.__getitem__(self, n % 4)