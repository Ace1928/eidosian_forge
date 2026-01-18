import collections
import os
import re
import sys
import functools
import itertools
def _unknown_as_blank(val):
    return '' if val == 'unknown' else val