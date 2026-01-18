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
def _ltrim_indent(indent, remove_width):
    width = 0
    for i, c in enumerate(indent):
        if width == remove_width:
            break
        if c == ' ':
            width += 1
        elif c == '\t':
            if width + 8 - width % 8 <= remove_width:
                width += 8 - width % 8
            else:
                return ' ' * (width + 8 - remove_width) + indent[i + 1:]
    return indent[i:]