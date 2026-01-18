from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceNetworkPassThroughLbTrafficPolicy(_messages.Message):
    """A BackendServiceNetworkPassThroughLbTrafficPolicy object.

  Fields:
    zonalAffinity: When configured, new connections are load balanced across
      healthy backend endpoints in the local zone.
  """
    zonalAffinity = _messages.MessageField('BackendServiceNetworkPassThroughLbTrafficPolicyZonalAffinity', 1)