from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZonePrivateVisibilityConfigNetwork(_messages.Message):
    """A ManagedZonePrivateVisibilityConfigNetwork object.

  Fields:
    kind: A string attribute.
    networkUrl: The fully qualified URL of the VPC network to bind to. Format
      this URL like https://www.googleapis.com/compute/v1/projects/{project}/g
      lobal/networks/{network}
  """
    kind = _messages.StringField(1, default='dns#managedZonePrivateVisibilityConfigNetwork')
    networkUrl = _messages.StringField(2)