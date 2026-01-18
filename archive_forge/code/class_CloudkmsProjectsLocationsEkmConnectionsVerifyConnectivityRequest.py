from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsEkmConnectionsVerifyConnectivityRequest(_messages.Message):
    """A CloudkmsProjectsLocationsEkmConnectionsVerifyConnectivityRequest
  object.

  Fields:
    name: Required. The name of the EkmConnection to verify.
  """
    name = _messages.StringField(1, required=True)