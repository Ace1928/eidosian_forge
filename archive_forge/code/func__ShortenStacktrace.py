from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def _ShortenStacktrace(stacktrace, url_encoded_length):
    """Cut out the middle entries of the stack trace to a given length.

  For instance:

  >>> stacktrace = '''
  ...   File "foo.py", line 10, in run
  ...     result = bar.run()
  ...   File "bar.py", line 11, in run
  ...     result = baz.run()
  ...   File "baz.py", line 12, in run
  ...     result = qux.run()
  ...   File "qux.py", line 13, in run
  ...     raise Exception(':(')
  ... '''
  >>> _ShortenStacktrace(stacktrace, 300) == '''  ...   File "foo.py", line 10, in run
  ...     result = bar.run()
  ...   [...]
  ...   File "baz.py", line 12, in run
  ...      result = qux.run()
  ...   File "qux.py", line 13, in run
  ...      raise Exception(':(')
  ... '''
  True


  Args:
    stacktrace: str, the stacktrace (might be formatted by _FormatTraceback)
        without the leading 'Traceback (most recent call last):' or 'Trace:'
    url_encoded_length: int, the length to shorten the stacktrace to (when
        URL-encoded).

  Returns:
    str, the shortened stacktrace.
  """
    stacktrace = stacktrace.strip('\n')
    lines = stacktrace.split('\n')
    entries = ['\n'.join(lines[i:i + STACKTRACE_LINES_PER_ENTRY]) for i in range(0, len(lines), STACKTRACE_LINES_PER_ENTRY)]
    if _UrlEncodeLen(stacktrace) <= url_encoded_length:
        return stacktrace + '\n'
    rest = entries[1:]
    while _UrlEncodeLen(_FormatStackTrace(entries[0], rest)) > url_encoded_length and len(rest) > 1:
        rest = rest[1:]
    return _FormatStackTrace(entries[0], rest)