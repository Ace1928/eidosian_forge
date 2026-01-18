from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesGetEffectiveFirewallsRequest(_messages.Message):
    """A ComputeInstancesGetEffectiveFirewallsRequest object.

  Fields:
    instance: Name of the instance scoping this request.
    networkInterface: The name of the network interface to get the effective
      firewalls.
    project: Project ID for this request.
    zone: The name of the zone for this request.
  """
    instance = _messages.StringField(1, required=True)
    networkInterface = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)
    zone = _messages.StringField(4, required=True)