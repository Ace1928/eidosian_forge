import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class CompareAction(argparse.Action):

    def __call__(self, parser, namespace, value, option_string=None):
        atomCompare_name, bondCompare_name = compare_shortcuts[value]
        namespace.atomCompare = atomCompare_name
        namespace.bondCompare = bondCompare_name