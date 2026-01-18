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
def _FormatTraceback(traceback):
    """Compacts stack trace portion of traceback and extracts exception.

  Args:
    traceback: str, the original unformatted traceback

  Returns:
    tuple of (str, str) where the first str is the formatted stack trace and the
    second str is exception.
  """
    match = re.search(PARTITION_TRACEBACK_PATTERN, traceback)
    if not match:
        return (traceback, '')
    stacktrace = match.group('stacktrace')
    exception = match.group('exception')
    formatted_stacktrace = '\n'.join((line.strip() for line in stacktrace.splitlines()))
    formatted_stacktrace += '\n'
    stacktrace_files = re.findall('File "(.*)"', stacktrace)
    for path in stacktrace_files:
        formatted_stacktrace = formatted_stacktrace.replace(path, _StripPath(path))
    formatted_stacktrace = re.sub(TRACEBACK_ENTRY_REGEXP, _StacktraceEntryReplacement, formatted_stacktrace)
    formatted_stacktrace = formatted_stacktrace.replace('Traceback (most recent call last):\n', '')
    return (formatted_stacktrace, exception)