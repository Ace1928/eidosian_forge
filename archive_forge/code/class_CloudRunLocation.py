from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunLocation(_messages.Message):
    """Information specifying where to deploy a Cloud Run Service.

  Fields:
    location: Required. The location for the Cloud Run Service. Format must be
      `projects/{project}/locations/{location}`.
  """
    location = _messages.StringField(1)