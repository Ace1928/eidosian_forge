import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def DT_info(self):
    """
        Returns (first label, second label, flip)
        """
    labels = self.strand_labels
    over = labels[3] + 1 if self.sign == 1 else labels[1] + 1
    under = labels[0] + 1
    if self.sign == 1:
        flip = 1 if labels[0] < labels[3] else 0
    else:
        flip = 0 if labels[0] < labels[1] else 1
    return (under, -over, flip) if over % 2 == 0 else (over, under, flip)