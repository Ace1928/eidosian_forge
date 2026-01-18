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
def _comp_exp(self, node, open_brace=None, close_brace=None):
    if open_brace:
        self.attr(node, 'compexp_open', [open_brace, self.ws], default=open_brace)
    self.visit(node.elt)
    for i, comp in enumerate(node.generators):
        self.visit(comp)
    if close_brace:
        self.attr(node, 'compexp_close', [self.ws, close_brace], default=close_brace)