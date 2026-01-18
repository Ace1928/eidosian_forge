from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _SplitInTwo(self, line):
    """Splits line into before and after, len(before) < self._max_index.

    Args:
      line: str, The line to split.

    Returns:
      (before, after)
        The line split into two parts. <before> is a list of strings that forms
        the first line of the split and <after> is a string containing the
        remainder of the line to split. The display width of <before> is
        < self._max_index. <before> contains the separator chars, including a
        newline.
    """
    punct_index = 0
    quoted_space_index = 0
    quoted_space_quote = None
    space_index = 0
    space_flag = False
    i = 0
    while i < self._max_index:
        c = line[i]
        i += 1
        if c == self._quote_char:
            self._quote_char = None
        elif self._quote_char:
            if c == ' ':
                quoted_space_index = i - 1
                quoted_space_quote = self._quote_char
        elif c in ('"', "'"):
            self._quote_char = c
            self._quote_index = i
            quoted_space_index = 0
        elif c == '\\':
            i += 1
        elif i < self._max_index:
            if c == ' ':
                if line[i] == '-':
                    space_flag = True
                    space_index = i
                elif space_flag:
                    space_flag = False
                else:
                    space_index = i
            elif c in (',', ';', '/', '|'):
                punct_index = i
            elif c == '=':
                space_flag = False
    separator = '\\\n'
    indent = _FIRST_INDENT
    if space_index:
        split_index = space_index
        indent = _SUBSEQUENT_INDENT
    elif quoted_space_index:
        split_index = quoted_space_index
        if quoted_space_quote == "'":
            separator = '\n'
        else:
            split_index += 1
    elif punct_index:
        split_index = punct_index
    else:
        split_index = self._max_index
    if split_index <= self._quote_index:
        self._quote_char = None
    else:
        self._quote_index = 0
    self._max_index = _SPLIT - _SECTION_INDENT - indent
    return ([line[:split_index], separator, ' ' * indent], line[split_index:])