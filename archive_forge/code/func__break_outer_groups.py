from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn
from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY
from typing import Dict
def _break_outer_groups(self):
    while self.max_width < self.output_width + self.buffer_width:
        group = self.group_queue.deq()
        if not group:
            return
        self._break_one_group(group)