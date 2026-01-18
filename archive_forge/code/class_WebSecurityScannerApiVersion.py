from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
class WebSecurityScannerApiVersion(object):
    """An enum representing the API version of Web Security Scanner.

  The WebSecurityScanner API version controls which version of WSS API to use
  for a certain command under certain release track.
  """

    class _VERSION(object):
        """An enum representing the API version of Web Security Manager."""

        def __init__(self, id, help_tag, help_note):
            self.id = id
            self.help_tag = help_tag
            self.help_note = help_note

        def __str__(self):
            return self.id

        def __eq__(self, other):
            return self.id == other.id
    V1BETA = _VERSION('v1beta', None, None)
    _ALL = V1BETA