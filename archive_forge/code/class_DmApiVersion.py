from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class DmApiVersion(object):
    """An enum representing the API version of Deployment Manager.

  The DM API version controls which version of DM API to use for a certain
  command under certain release track.
  """

    class _VERSION(object):
        """An enum representing the API version of Deployment Manager."""

        def __init__(self, id, help_tag, help_note):
            self.id = id
            self.help_tag = help_tag
            self.help_note = help_note

        def __str__(self):
            return self.id

        def __eq__(self, other):
            return self.id == other.id
    V2 = _VERSION('v2', None, None)
    ALPHA = _VERSION('alpha', '{0}(ALPHA){0} '.format('*'), 'The DM API currently used is ALPHA and may change without notice.')
    V2BETA = _VERSION('v2beta', '{0}(V2BETA){0} '.format('*'), 'The DM API currently used is V2BETA and may change without notice.')
    _ALL = (V2, ALPHA, V2BETA)