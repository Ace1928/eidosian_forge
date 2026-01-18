import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
class _MockTransform(object):

    def __init__(self):
        self._sum = 0
        self._memorize_chunk_called = 0
        self._memorize_finish_called = 0

    def memorize_chunk(self, data):
        self._memorize_chunk_called += 1
        import numpy as np
        self._sum += np.sum(data)

    def memorize_finish(self):
        self._memorize_finish_called += 1

    def transform(self, data):
        return data - self._sum