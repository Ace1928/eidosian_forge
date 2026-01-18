import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
def _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy):
    """
    Extends line with nicely formatted (possibly multi-line) string ``word``.
    """
    words = word.splitlines()
    if len(words) == 1 or legacy <= 113:
        return _extendLine(s, line, word, line_width, next_line_prefix, legacy)
    max_word_length = max((len(word) for word in words))
    if len(line) + max_word_length > line_width and len(line) > len(next_line_prefix):
        s += line.rstrip() + '\n'
        line = next_line_prefix + words[0]
        indent = next_line_prefix
    else:
        indent = len(line) * ' '
        line += words[0]
    for word in words[1:]:
        s += line.rstrip() + '\n'
        line = indent + word
    suffix_length = max_word_length - len(words[-1])
    line += suffix_length * ' '
    return (s, line)