import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _rng_html_rewrite(func):
    """Rewrite the HTML rendering of ``np.random.default_rng``.

    This is intended to decorate
    ``numpydoc.docscrape_sphinx.SphinxDocString._str_examples``.

    Examples are only run by Sphinx when there are plot involved. Even so,
    it does not change the result values getting printed.
    """
    pattern = re.compile('np.random.default_rng\\((0x[0-9A-F]+|\\d+)\\)', re.I)

    def _wrapped(*args, **kwargs):
        res = func(*args, **kwargs)
        lines = [re.sub(pattern, 'np.random.default_rng()', line) for line in res]
        return lines
    return _wrapped