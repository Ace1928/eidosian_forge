from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from contextlib import closing
import copy
import inspect
import os
import re
import sys
import textwrap
from io import StringIO
import numba.core.dispatcher
from numba.core import ir
def _getindent(text):
    m = re_longest_white_prefix.match(text)
    if not m:
        return ''
    else:
        return ' ' * len(m.group(0))