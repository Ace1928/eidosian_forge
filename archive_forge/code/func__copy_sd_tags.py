import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _copy_sd_tags(mol, fragment):
    fragment.SetProp('_Name', mol.GetProp('_Name'))
    for name in mol.GetPropNames():
        if name.startswith('_'):
            continue
        fragment.SetProp(name, mol.GetProp(name))