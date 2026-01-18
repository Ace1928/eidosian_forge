from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys
from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator
def _one_of_symbols():
    next_token = self.tokens.next()
    found = next((s for s in symbols if s == next_token.src), None)
    if found is None:
        raise AnnotationError('Expected one of: %r, but found: %r' % (symbols, next_token.src))
    return found