import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
class EDistance(IntEnum):
    DotProduct = 0
    L1 = 1
    L2Sqr = 2