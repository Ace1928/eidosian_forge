from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import difflib
import enum
import io
import re
import sys
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import util as format_util
import six
def ExtractHelpStrings(docstring):
    """Extracts short help and long help from a docstring.

  If the docstring contains a blank line (i.e., a line consisting of zero or
  more spaces), everything before the first blank line is taken as the short
  help string and everything after it is taken as the long help string. The
  short help is flowing text with no line breaks, while the long help may
  consist of multiple lines, each line beginning with an amount of whitespace
  determined by dedenting the docstring.

  If the docstring does not contain a blank line, the sequence of words in the
  docstring is used as both the short help and the long help.

  Corner cases: If the first line of the docstring is empty, everything
  following it forms the long help, and the sequence of words of in the long
  help (without line breaks) is used as the short help. If the short help
  consists of zero or more spaces, None is used instead. If the long help
  consists of zero or more spaces, the short help (which might or might not be
  None) is used instead.

  Args:
    docstring: The docstring from which short and long help are to be taken

  Returns:
    a tuple consisting of a short help string and a long help string

  """
    if docstring:
        unstripped_doc_lines = docstring.splitlines()
        stripped_doc_lines = [s.strip() for s in unstripped_doc_lines]
        try:
            empty_line_index = stripped_doc_lines.index('')
            short_help = ' '.join(stripped_doc_lines[:empty_line_index])
            raw_long_help = '\n'.join(unstripped_doc_lines[empty_line_index + 1:])
            long_help = textwrap.dedent(raw_long_help).strip()
        except ValueError:
            short_help = ' '.join(stripped_doc_lines).strip()
            long_help = ''
        if not short_help:
            short_help = ' '.join(stripped_doc_lines[empty_line_index + 1:]).strip()
        return (short_help, long_help or short_help)
    else:
        return ('', '')