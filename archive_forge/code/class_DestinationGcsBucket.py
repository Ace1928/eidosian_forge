from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationGcsBucket(_messages.Message):
    """Google Cloud Storage as a destination.

  Fields:
    uri: Required. URI to a Cloud Storage object in format: 'gs:///'.
  """
    uri = _messages.StringField(1)