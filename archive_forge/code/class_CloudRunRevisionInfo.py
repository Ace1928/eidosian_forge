from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunRevisionInfo(_messages.Message):
    """For display only. Metadata associated with a Cloud Run revision.

  Fields:
    displayName: Name of a Cloud Run revision.
    location: Location in which this revision is deployed.
    serviceUri: URI of Cloud Run service this revision belongs to.
    uri: URI of a Cloud Run revision.
  """
    displayName = _messages.StringField(1)
    location = _messages.StringField(2)
    serviceUri = _messages.StringField(3)
    uri = _messages.StringField(4)