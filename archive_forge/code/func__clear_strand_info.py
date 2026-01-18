import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _clear_strand_info(self):
    self.strand_labels = CyclicList4()
    self.strand_components = CyclicList4()