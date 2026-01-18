from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def ValidFilename(filename):
    """Determines if a file name is valid.

  Args:
    filename: The file name to validate. The file name must be a valid file
        name:
            - It must only contain letters, numbers, and the following special
              characters:  `@`, `_`, `+`, `/` `$`, `.`, `-`, or '~'.
            - It must be less than 256 characters.
            - It must not contain `/./`, `/../`, or `//`.
            - It must not end in `/`.
            - All spaces must be in the middle of a directory or file name.

  Returns:
    An error string if the file name is invalid. `''` is returned if the file
    name is valid.
  """
    if not filename:
        return 'Filename cannot be empty'
    if len(filename) > 1024:
        return 'Filename cannot exceed 1024 characters: %s' % filename
    if _file_path_negative_1_re.search(filename) is not None:
        return 'Filename cannot contain "." or ".." or start with "-" or "_ah/": %s' % filename
    if _file_path_negative_2_re.search(filename) is not None:
        return 'Filename cannot have trailing / or contain //: %s' % filename
    if _file_path_negative_3_re.search(filename) is not None:
        return 'Any spaces must be in the middle of a filename: %s' % filename
    return ''