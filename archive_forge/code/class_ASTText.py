import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
class ASTText(ASTTextBase, object):
    """
  Supports the same ``get_text*`` methods as ``ASTTokens``,
  but uses the AST to determine the text positions instead of tokens.
  This is faster than ``ASTTokens`` as it requires less setup work.

  It also (sometimes) supports nodes inside f-strings, which ``ASTTokens`` doesn't.

  Some node types and/or Python versions are not supported.
  In these cases the ``get_text*`` methods will fall back to using ``ASTTokens``
  which incurs the usual setup cost the first time.
  If you want to avoid this, check ``supports_tokenless(node)`` before calling ``get_text*`` methods.
  """

    def __init__(self, source_text, tree=None, filename='<unknown>'):
        super(ASTText, self).__init__(source_text, filename)
        self._tree = tree
        if self._tree is not None:
            annotate_fstring_nodes(self._tree)
        self._asttokens = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = ast.parse(self._text, self._filename)
            annotate_fstring_nodes(self._tree)
        return self._tree

    @property
    def asttokens(self):
        if self._asttokens is None:
            self._asttokens = ASTTokens(self._text, tree=self.tree, filename=self._filename)
        return self._asttokens

    def _get_text_positions_tokenless(self, node, padded):
        """
    Version of ``get_text_positions()`` that doesn't use tokens.
    """
        if sys.version_info[:2] < (3, 8):
            raise AssertionError('This method should only be called internally after checking supports_tokenless()')
        if is_module(node):
            return ((1, 0), self._line_numbers.offset_to_line(len(self._text)))
        if getattr(node, 'lineno', None) is None:
            return ((1, 0), (1, 0))
        assert node
        decorators = getattr(node, 'decorator_list', [])
        if not decorators:
            decorators_node = getattr(node, 'decorators', None)
            decorators = getattr(decorators_node, 'nodes', [])
        if decorators:
            start_node = decorators[0]
        else:
            start_node = node
        start_lineno = start_node.lineno
        end_node = last_stmt(node)
        if padded and (start_lineno != end_node.lineno or (start_lineno != node.end_lineno and getattr(node, 'doc_node', None) and is_stmt(node))):
            start_col_offset = 0
        else:
            start_col_offset = self._line_numbers.from_utf8_col(start_lineno, start_node.col_offset)
        start = (start_lineno, start_col_offset)
        end_lineno = cast(int, end_node.end_lineno)
        end_col_offset = cast(int, end_node.end_col_offset)
        end_col_offset = self._line_numbers.from_utf8_col(end_lineno, end_col_offset)
        end = (end_lineno, end_col_offset)
        return (start, end)

    def get_text_positions(self, node, padded):
        """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
        if getattr(node, '_broken_positions', None):
            return ((1, 0), (1, 0))
        if supports_tokenless(node):
            return self._get_text_positions_tokenless(node, padded)
        return self.asttokens.get_text_positions(node, padded)