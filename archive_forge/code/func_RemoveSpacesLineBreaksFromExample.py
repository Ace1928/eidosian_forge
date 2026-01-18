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
def RemoveSpacesLineBreaksFromExample(example):
    """Returns the example with redundant spaces and line breaks removed.

  If a character sequence is quoted (either single or double quote), we will
  not touch its value. Single quote is not allowed within single quote even
  with a preceding backslash. Double quote is allowed in double quote with
  preceding backslash though. If the spaces and line breaks are within quote,
  they are not touched.

  Args:
    example, str: Example line to process.
  """
    res = []
    example = example.strip()
    pos = 0
    while pos < len(example):
        c = example[pos]
        if c not in ['"', "'"]:
            if c == '\n':
                c = ' '
            if not (c == ' ' and res and (res[-1] == ' ')):
                res.append(c)
            pos += 1
        elif c == "'":
            res.append(c)
            pos += 1
            while pos < len(example) and example[pos] != "'":
                res.append(example[pos])
                pos += 1
            if pos < len(example):
                res.append(example[pos])
                pos += 1
        else:
            res.append(example[pos])
            pos += 1
            while pos < len(example) and (not (example[pos] == '"' and _PrecedingBackslashCount(res) % 2 == 0)):
                res.append(example[pos])
                pos += 1
            if pos < len(example):
                res.append(example[pos])
                pos += 1
    return ''.join(res)