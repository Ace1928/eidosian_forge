import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
def _print_summary(self):
    max_cost = 0
    for entry in self.data:
        totaltime = int(entry.totaltime * 1000)
        max_cost = max(max_cost, totaltime)
    self.out_file.write('summary: %d\n' % (max_cost,))