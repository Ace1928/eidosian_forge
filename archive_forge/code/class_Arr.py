import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
class Arr(list):

    def __getitem__(self, s):
        if isinstance(s, slice):
            self._start = s.start
            return self
        return super().__getitem__(s)

    def __getslice__(self, i, j):
        self._start = i
        return self

    def __iadd__(self, i):
        for j in range(self._start, len(self)):
            self[j] += i