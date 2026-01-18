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
def _get_indent_diff(outer, inner):
    """Computes the whitespace added to an indented block.

  Finds the portion of an indent prefix that is added onto the outer indent. In
  most cases, the inner indent starts with the outer indent, but this is not
  necessarily true. For example, the outer block could be indented to four
  spaces and its body indented with one tab (effectively 8 spaces).

  Arguments:
    outer: (string) Indentation of the outer block.
    inner: (string) Indentation of the inner block.
  Returns:
    The string whitespace which is added to the indentation level when moving
    from outer to inner.
  """
    outer_w = _get_indent_width(outer)
    inner_w = _get_indent_width(inner)
    diff_w = inner_w - outer_w
    if diff_w <= 0:
        return None
    return _ltrim_indent(inner, inner_w - diff_w)