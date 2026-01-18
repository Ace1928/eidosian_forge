import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def _randomizedPartition(array, begin, end):
    i = random.randint(begin, end)
    array[begin], array[i] = (array[i], array[begin])
    return _partition(array, begin, end)