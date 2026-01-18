from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
def _FormatNonAsciiMarkerString(args):
    """Format a string that will mark the first non-ASCII character it contains.


  Example:

  >>> args = ['command.py', '--foo=\\xce\\x94']
  >>> _FormatNonAsciiMarkerString(args) == (
  ...     'command.py --foo=\\u0394\\n'
  ...     '                 ^ invalid character'
  ... )
  True

  Args:
    args: The arg list for the command executed

  Returns:
    unicode, a properly formatted string with two lines, the second of which
      indicates the non-ASCII character in the first.

  Raises:
    ValueError: if the given string is all ASCII characters
  """
    pos = 0
    for arg in args:
        first_non_ascii_index = _NonAsciiIndex(arg)
        if first_non_ascii_index >= 0:
            pos += first_non_ascii_index
            break
        pos += len(arg) + 1
    else:
        raise ValueError('The command line is composed entirely of ASCII characters.')
    marker_string = ' ' * pos + _MARKER
    align = len(marker_string)
    args_string = ' '.join([console_attr.SafeText(arg) for arg in args])
    width, _ = console_attr_os.GetTermSize()
    fill = '...'
    if width < len(_MARKER) + len(fill):
        return '\n'.join((args_string, marker_string))
    formatted_args_string = _TruncateToLineWidth(args_string.ljust(align), align, width, fill=fill).rstrip()
    formatted_marker_string = _TruncateToLineWidth(marker_string, align, width)
    return '\n'.join((formatted_args_string, formatted_marker_string))