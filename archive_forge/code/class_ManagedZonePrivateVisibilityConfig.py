from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZonePrivateVisibilityConfig(_messages.Message):
    """A ManagedZonePrivateVisibilityConfig object.

  Fields:
    gkeClusters: The list of Google Kubernetes Engine clusters that can see
      this zone.
    kind: A string attribute.
    networks: The list of VPC networks that can see this zone.
  """
    gkeClusters = _messages.MessageField('ManagedZonePrivateVisibilityConfigGKECluster', 1, repeated=True)
    kind = _messages.StringField(2, default='dns#managedZonePrivateVisibilityConfig')
    networks = _messages.MessageField('ManagedZonePrivateVisibilityConfigNetwork', 3, repeated=True)