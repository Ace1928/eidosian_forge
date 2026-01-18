import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
class GrouperView:
    """Immutable view over a `.Grouper`."""

    def __init__(self, grouper):
        self._grouper = grouper

    def __contains__(self, item):
        return item in self._grouper

    def __iter__(self):
        return iter(self._grouper)

    def joined(self, a, b):
        return self._grouper.joined(a, b)

    def get_siblings(self, a):
        return self._grouper.get_siblings(a)