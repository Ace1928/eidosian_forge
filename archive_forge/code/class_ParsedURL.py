from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class ParsedURL(object):
    """Dispath Entry URL holder class.

  Attributes:
    host_pattern: The host pattern component of the URL pattern.
    host_exact: True if the host pattern does not start with a *.
    host: host_pattern  with any leading * removed.
    path_pattern: The path pattern component of the URL pattern.
    path_exact: True if the path_pattern does not end with a *.
    path: path_pattern with any trailing * removed.
  """

    def __init__(self, url_pattern):
        """Initializes this ParsedURL with an URL pattern value.

    Args:
      url_pattern: An URL pattern that conforms to the regular expression
          '^([^/]+)(/.*)$'.

    Raises:
      validation.ValidationError: When url_pattern does not match the required
          regular expression.
    """
        split_matcher = _ValidateMatch(_URL_SPLITTER_RE, url_pattern, "invalid url '%s'" % url_pattern)
        self.host_pattern, self.path_pattern = split_matcher.groups()
        if self.host_pattern.startswith('*'):
            self.host_exact = False
            self.host = self.host_pattern[1:]
        else:
            self.host_exact = True
            self.host = self.host_pattern
        if self.path_pattern.endswith('*'):
            self.path_exact = False
            self.path = self.path_pattern[:-1]
        else:
            self.path_exact = True
            self.path = self.path_pattern