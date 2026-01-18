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
def _UrlTruncateLines(string, url_encoded_length):
    """Truncates the given string to the given URL-encoded length.

  Always cuts at a newline.

  Args:
    string: str, the string to truncate
    url_encoded_length: str, the length to which to truncate

  Returns:
    tuple of (str, str), where the first str is the truncated version of the
    original string and the second str is the remainder.
  """
    lines = string.split('\n')
    included_lines = []
    excluded_lines = []
    max_str_len = url_encoded_length - _UrlEncodeLen(TRUNCATED_INFO_MESSAGE + '\n')
    while lines and _UrlEncodeLen('\n'.join(included_lines + lines[:1])) <= max_str_len:
        included_lines.append(lines.pop(0))
    excluded_lines = lines
    if excluded_lines:
        included_lines.append(TRUNCATED_INFO_MESSAGE)
    return ('\n'.join(included_lines), '\n'.join(excluded_lines))