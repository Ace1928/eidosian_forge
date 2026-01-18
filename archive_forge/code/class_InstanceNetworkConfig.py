from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceNetworkConfig(_messages.Message):
    """Metadata related to instance level network configuration.

  Fields:
    authorizedExternalNetworks: Optional. A list of external network
      authorized to access this instance.
    enablePublicIp: Optional. Enabling public ip for the instance.
  """
    authorizedExternalNetworks = _messages.MessageField('AuthorizedNetwork', 1, repeated=True)
    enablePublicIp = _messages.BooleanField(2)