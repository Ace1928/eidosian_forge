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
def _optional_token(self, token_type, token_val):
    token = self.tokens.peek()
    if not token or token.type != token_type or token.src != token_val:
        return ''
    else:
        self.tokens.next()
        return token.src + self.ws()